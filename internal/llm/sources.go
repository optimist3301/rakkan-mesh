package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sync"

	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/optimist3301/rakkan-mesh/internal/blob"
	"github.com/optimist3301/rakkan-mesh/internal/llm/dsv41"
)

// Files of Hub models on a node (hub.go): kept as blobs, named by their
// content (the pinned hash), and offered to other members in the adverts.
// A file is had from the blob store if there, else from a member offering
// it, else from the Hub; whatever the source, its bytes are hashed and
// must be the pinned ones. A download goes on in the background when the
// request that started it gives up: a 100 GiB file outlasts a stage's
// opening, and the next request finds it.

type sources struct {
	mu       sync.Mutex
	have     map[string]blob.ID // by HubFile.key
	path     string             // where have is kept, "" for nowhere
	inflight map[string]*sourceFetch
}

type sourceFetch struct {
	done chan struct{}
	id   blob.ID
	err  error
}

func newSources(dir string) *sources {
	s := &sources{have: map[string]blob.ID{}, inflight: map[string]*sourceFetch{}}
	if dir != "" {
		s.path = filepath.Join(dir, "hub-files.json")
		if b, err := os.ReadFile(s.path); err == nil {
			var m map[string]string
			if json.Unmarshal(b, &m) == nil {
				for k, v := range m {
					if id, err := blob.ParseID(v); err == nil {
						s.have[k] = id
					}
				}
			}
		}
	}
	return s
}

// record notes a file had, and keeps the list. The caller holds s.mu.
func (s *sources) record(key string, id blob.ID) {
	s.have[key] = id
	if s.path == "" {
		return
	}
	m := map[string]string{}
	for k, v := range s.have {
		m[k] = v.String()
	}
	b, _ := json.Marshal(m)
	tmp := s.path + ".tmp"
	if os.WriteFile(tmp, b, 0o600) == nil {
		os.Rename(tmp, s.path)
	}
}

// offered lists the files this node has, for its adverts.
func (s *Service) offeredFiles() map[string]string {
	s.src.mu.Lock()
	defer s.src.mu.Unlock()
	out := map[string]string{}
	for k, id := range s.src.have {
		if s.blobs.Store().Has(id) {
			out[k] = id.String()
		}
	}
	return out
}

// hubFile makes a file of a Hub model available as a blob and returns it.
func (s *Service) hubFile(ctx context.Context, o *HubOrigin, name string) (blob.ID, error) {
	f, ok := o.Files[name]
	if !ok {
		return blob.ID{}, fmt.Errorf("hub: %s is not pinned", name)
	}
	key := f.key()
	s.src.mu.Lock()
	if id, ok := s.src.have[key]; ok && s.blobs.Store().Has(id) {
		s.src.mu.Unlock()
		return id, nil
	}
	sf, ok := s.src.inflight[key]
	if !ok {
		sf = &sourceFetch{done: make(chan struct{})}
		s.src.inflight[key] = sf
		s.wg.Go(func() {
			// The service's context, not the request's: see above.
			sf.id, sf.err = s.fetchHubFile(s.ctx, o, name, f)
			s.src.mu.Lock()
			delete(s.src.inflight, key)
			if sf.err == nil {
				s.src.record(key, sf.id)
			}
			s.src.mu.Unlock()
			close(sf.done)
			s.notify()
		})
	}
	s.src.mu.Unlock()
	select {
	case <-sf.done:
		return sf.id, sf.err
	case <-ctx.Done():
		return blob.ID{}, fmt.Errorf("%s still downloading: %w", name, ctx.Err())
	}
}

func (s *Service) fetchHubFile(ctx context.Context, o *HubOrigin, name string, f HubFile) (blob.ID, error) {
	key := f.key()
	// From members offering it.
	for p, a := range s.servers() {
		idStr, ok := a.Files[key]
		if !ok {
			continue
		}
		id, err := blob.ParseID(idStr)
		if err != nil {
			continue
		}
		if err := s.blobs.Fetch(ctx, id, []peer.ID{p}, f.Size); err != nil {
			s.log.Info("hub file from a member failed", "file", name, "peer", p.ShortString(), "err", err)
			continue
		}
		if err := s.checkFile(id, f); err != nil {
			s.log.Warn("a member's hub file is not the pinned one", "file", name, "peer", p.ShortString(), "err", err)
			continue
		}
		s.log.Info("hub file from a member", "file", name, "peer", p.ShortString())
		return id, nil
	}
	if s.cfg.Hub == nil {
		return blob.ID{}, fmt.Errorf("%s: no member offers it, and downloads from the Hub are off", name)
	}
	s.log.Info("downloading from the Hub", "repo", o.Repo, "file", name, "bytes", f.Size)
	r := s.cfg.Hub.open(ctx, o.Repo, o.Revision, name, f)
	id, m, err := s.blobs.Store().Put(r)
	if err != nil {
		return blob.ID{}, fmt.Errorf("%s: %w", name, err)
	}
	if !r.(*hubReader).verified || m.Size != f.Size {
		return blob.ID{}, fmt.Errorf("%s: download incomplete", name)
	}
	return id, nil
}

// checkFile hashes a stored blob against a file's pin.
func (s *Service) checkFile(id blob.ID, f HubFile) error {
	r, err := s.blobs.Store().Open(id)
	if err != nil {
		return err
	}
	defer r.Close()
	if r.Size() != f.Size {
		return fmt.Errorf("%d bytes, pinned %d", r.Size(), f.Size)
	}
	h := f.hasher()
	if _, err := io.Copy(h, r); err != nil {
		return err
	}
	if !f.check(h) {
		return errors.New("hash differs from the pinned one")
	}
	return nil
}

// hubTensors opens the files of a part and lists their tensors; close
// releases them (the Engram tables read in place need them open).
func (s *Service) hubTensors(ctx context.Context, o *HubOrigin, names []string) (map[string]*stTensor, func(), error) {
	var readers []*blob.Reader
	closer := func() {
		for _, r := range readers {
			r.Close()
		}
	}
	tensors := map[string]*stTensor{}
	for _, name := range names {
		id, err := s.hubFile(ctx, o, name)
		if err != nil {
			closer()
			return nil, nil, err
		}
		r, err := s.blobs.Store().Open(id)
		if err != nil {
			closer()
			return nil, nil, err
		}
		readers = append(readers, r)
		if err := readSafetensors(&lockedReaderAt{r: r}, tensors); err != nil {
			closer()
			return nil, nil, fmt.Errorf("%s: %w", name, err)
		}
	}
	return tensors, closer, nil
}

func (l *lockedReaderAt) Size() int64 { return l.r.Size() }

// dsTensor reads a tensor of a source file into memory.
func dsTensor(name string, t *stTensor) (*dsv41.Tensor, error) {
	data := make([]byte, t.Offsets[1]-t.Offsets[0])
	if _, err := t.file.ReadAt(data, t.base+t.Offsets[0]); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return dsv41.NewTensor(t.DType, t.Shape, data)
}

// loadHubPart builds blocks [from, to) of a Hub model from its files,
// the Engram tables read from them in place.
func (s *Service) loadHubPart(ctx context.Context, m *Manifest, from, to int) (*dsv41.Model, func(), error) {
	o := m.Hub
	var names []string
	for i := from; i < to; i++ {
		for _, n := range o.Layers[i] {
			if !slices.Contains(names, n) {
				names = append(names, n)
			}
		}
	}
	all, closer, err := s.hubTensors(ctx, o, names)
	if err != nil {
		return nil, nil, err
	}
	tensors := map[string]*dsv41.Tensor{}
	engram := map[int]dsv41.EngramTable{}
	for name, t := range all {
		l, ok := dsv41.TensorLayer(name)
		if !ok || l < from || l >= to {
			continue
		}
		p := fmt.Sprintf("layers.%d.engram.embed.", l)
		switch name {
		case p + "weight", p + "scale":
			e := engram[l]
			sec := io.NewSectionReader(t.file, t.base+t.Offsets[0], t.Offsets[1]-t.Offsets[0])
			if name == p+"weight" {
				e.Rows = sec
			} else {
				e.Scales = sec
			}
			engram[l] = e
			continue
		}
		tt, err := dsTensor(name, t)
		if err != nil {
			closer()
			return nil, nil, err
		}
		tensors[name] = tt
	}
	part, err := dsv41.NewPart(m.ds, tensors, from, to, engram)
	if err != nil {
		closer()
		return nil, nil, err
	}
	return part, closer, nil
}

// hubDiskBytes is what layer i's files take on disk.
func hubDiskBytes(o *HubOrigin, i int) int64 {
	var n int64
	for _, name := range o.Layers[i] {
		n += o.Files[name].Size
	}
	return n
}

// ImportHub imports a DeepSeek-V4.1 model from the Hub (see the function
// ImportHub) with the service's Hub client.
func (s *Service) ImportHub(ctx context.Context, repo, revision, name string) (blob.ID, *Manifest, error) {
	if s.cfg.Hub == nil {
		return blob.ID{}, nil, errors.New("downloads from the Hugging Face Hub are off on this node")
	}
	return ImportHub(ctx, s.blobs.Store(), s.cfg.Hub, repo, revision, name)
}

// Prefetch gets this node the files of a Hub model's layers [from, to),
// all if to is 0, and reports their bytes. Downloads go on if ctx ends
// first.
func (s *Service) Prefetch(ctx context.Context, model blob.ID, from, to int) (int64, error) {
	m, err := s.fetchManifest(ctx, model, nil)
	if err != nil {
		return 0, err
	}
	o := m.Hub
	if o == nil {
		return 0, errors.New("not a model from the Hub: its parts are fetched from members as needed")
	}
	if to == 0 {
		to = len(o.Layers)
	}
	if from < 0 || to > len(o.Layers) || from >= to {
		return 0, fmt.Errorf("layers %d..%d outside the model's %d", from, to, len(o.Layers))
	}
	var names []string
	for i := from; i < to; i++ {
		for _, n := range o.Layers[i] {
			if !slices.Contains(names, n) {
				names = append(names, n)
			}
		}
	}
	var total int64
	for _, n := range names {
		if _, err := s.hubFile(ctx, o, n); err != nil {
			return total, err
		}
		total += o.Files[n].Size
	}
	return total, nil
}
