// Package gpu runs the matrix-vector products of language models on GPUs
// through Vulkan compute, which every current GPU vendor supports (NVIDIA,
// AMD, Intel, Apple through MoltenVK, mobile GPUs), as do software
// implementations such as Mesa's lavapipe. The kernels reproduce package
// num's arithmetic bit for bit (see kernels.go), so GPU and CPU nodes are
// interchangeable within one pipeline.
package gpu

import (
	"encoding/binary"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"sync"
	"unsafe"

	"github.com/optimist3301/rakkan-mesh/internal/num"
)

// Info describes a GPU.
type Info struct {
	Index int    `json:"index"`
	Name  string `json:"name"`
	// Type is "discrete", "integrated", "virtual", "cpu" (a software
	// implementation) or "other".
	Type string `json:"type"`
	// Memory is the largest device-local memory heap, in bytes.
	Memory int64  `json:"memory"`
	API    string `json:"api"`
	// Exact reports whether the device guarantees the IEEE behaviour the
	// kernels need to match the CPU bit for bit: denormals preserved and
	// round-to-nearest-even (or a CPU implementation, which has them).
	Exact bool `json:"exact"`
	// Denormals reports whether the device promises to keep denormal
	// floats. Without, float32 arithmetic whose operands or results are
	// denormal may differ from the CPU's (llvmpipe says it does not keep
	// them); DeepSeek-V4.1's kernels meet them only with extreme values.
	Denormals bool `json:"denormals"`

	pd       uintptr
	apiMajor uint32
	apiMinor uint32
	modes    floatModes
	maxRange uint64
}

type instance struct {
	handle uintptr
}

var (
	instOnce sync.Once
	vkInst   *instance
	instErr  error
)

func getInstance() (*instance, error) {
	instOnce.Do(func() {
		if instErr = load(); instErr != nil {
			return
		}
		var version uint32
		if r := vk.EnumerateInstanceVersion(&version); r != vkSuccess || version < apiVersion(1, 1) {
			instErr = fmt.Errorf("%w: Vulkan 1.1 or later needed", ErrNoVulkan)
			return
		}
		app := vkApplicationInfo{sType: stApplicationInfo, pApplicationName: cstr("rakkan"), pEngineName: cstr("rakkan"),
			apiVersion: min(version, apiVersion(1, 2))}
		info := vkInstanceCreateInfo{sType: stInstanceCreateInfo, pApplicationInfo: &app}
		// MoltenVK devices are listed only to applications that accept
		// portability (partially conformant) implementations.
		var names []*byte
		if hasInstanceExtension("VK_KHR_portability_enumeration") {
			names = append(names, cstr("VK_KHR_portability_enumeration"))
			info.flags = instanceEnumeratePortability
		}
		if len(names) > 0 {
			info.enabledExtensionCount, info.ppEnabledExtensionNames = uint32(len(names)), &names[0]
		}
		var h uintptr
		r := vk.CreateInstance(&info, nil, &h)
		runtime.KeepAlive(names)
		if instErr = check("create instance", r); instErr == nil {
			vkInst = &instance{handle: h}
		}
	})
	return vkInst, instErr
}

func hasInstanceExtension(name string) bool {
	var n uint32
	if vk.EnumerateInstanceExtensionProperties(nil, &n, nil) != vkSuccess || n == 0 {
		return false
	}
	buf := make([]byte, int(n)*260)
	if vk.EnumerateInstanceExtensionProperties(nil, &n, unsafe.Pointer(&buf[0])) != vkSuccess {
		return false
	}
	for i := range int(n) {
		if gostr(buf[i*260:i*260+256]) == name {
			return true
		}
	}
	return false
}

func deviceExtensions(pd uintptr) []string {
	var n uint32
	if vk.EnumerateDeviceExtensionProperties(pd, nil, &n, nil) != vkSuccess || n == 0 {
		return nil
	}
	buf := make([]byte, int(n)*260)
	if vk.EnumerateDeviceExtensionProperties(pd, nil, &n, unsafe.Pointer(&buf[0])) != vkSuccess {
		return nil
	}
	var out []string
	for i := range int(n) {
		out = append(out, gostr(buf[i*260:i*260+256]))
	}
	return out
}

// Devices lists the GPUs Vulkan offers, software implementations included.
func Devices() ([]Info, error) {
	in, err := getInstance()
	if err != nil {
		return nil, err
	}
	var n uint32
	if err := check("enumerate devices", vk.EnumeratePhysicalDevices(in.handle, &n, nil)); err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}
	pds := make([]uintptr, n)
	if err := check("enumerate devices", vk.EnumeratePhysicalDevices(in.handle, &n, &pds[0])); err != nil {
		return nil, err
	}
	var out []Info
	for i, pd := range pds[:n] {
		fc := vkFloatControls{sType: stFloatControlsProperties}
		p2 := vkProperties2{sType: stPhysicalDeviceProps2, pNext: unsafe.Pointer(&fc)}
		vk.GetPhysicalDeviceProperties2(pd, &p2)
		props := p2.props[:]
		api := binary.LittleEndian.Uint32(props[0:])
		info := Info{Index: i, pd: pd, Name: gostr(props[20:276]), apiMajor: api >> 22 & 0x7f, apiMinor: api >> 12 & 0x3ff}
		info.API = fmt.Sprintf("%d.%d", info.apiMajor, info.apiMinor)
		// VkPhysicalDeviceLimits starts at 296; maxStorageBufferRange is
		// its eighth word.
		info.maxRange = uint64(binary.LittleEndian.Uint32(props[296+7*4:]))
		info.Type = map[uint32]string{physicalDeviceIntegrated: "integrated", physicalDeviceDiscrete: "discrete",
			physicalDeviceVirtual: "virtual", physicalDeviceCPU: "cpu"}[binary.LittleEndian.Uint32(props[16:])]
		if info.Type == "" {
			info.Type = "other"
		}
		if info.apiMajor > 1 || info.apiMinor >= 2 {
			info.modes = floatModes{signedZeroInfNan: fc.flags[1] != 0, denormPreserve: fc.flags[4] != 0, roundRTE: fc.flags[10] != 0}
		}
		info.Exact = info.Type == "cpu" || info.modes.denormPreserve && info.modes.roundRTE
		info.Denormals = info.modes.denormPreserve
		var mem [520]byte
		vk.GetPhysicalDeviceMemoryProperties(pd, &mem)
		heaps := binary.LittleEndian.Uint32(mem[260:])
		for h := range min(heaps, 16) {
			size := binary.LittleEndian.Uint64(mem[264+16*h:])
			flags := binary.LittleEndian.Uint32(mem[264+16*h+8:])
			if flags&1 != 0 && int64(size) > info.Memory {
				info.Memory = int64(size)
			}
		}
		out = append(out, info)
	}
	return out, nil
}

// Best picks the device to use by default: the discrete GPU with the most
// memory, else an integrated or virtual one, among those computing
// exactly (Info.Exact). Software implementations are never picked by
// default: the CPU kernels are faster.
func Best(devs []Info) (Info, bool) {
	rank := map[string]int{"discrete": 3, "integrated": 2, "virtual": 1}
	best, ok := Info{}, false
	for _, d := range devs {
		if rank[d.Type] == 0 || !d.Exact {
			continue
		}
		if !ok || rank[d.Type] > rank[best.Type] || rank[d.Type] == rank[best.Type] && d.Memory > best.Memory {
			best, ok = d, true
		}
	}
	return best, ok
}

// The widest input and output a model may have.
const (
	maxCols = 1 << 18 // Config.Hidden's limit
	maxRows = 1 << 20 // Config.Vocab's limit
)

// Device is an opened GPU. Its methods are safe for concurrent use; the
// submissions run one at a time.
type Device struct {
	info   Info
	dev    uintptr
	queue  uintptr
	family uint32
	memory [520]byte

	pool  uint64
	cmd   uintptr
	fence uint64

	setLayouts [2]uint64 // set 0: weights or vectors; set 1: an arena
	layout     uint64
	kernels    [kindAllCount]*pipeline // DeepSeek's built on first use
	setPool    uint64                  // descriptor sets of arenas and vectors

	host *Arena // for MulVecs: input, output and quantized input

	mu   sync.Mutex
	used int64 // bytes of matrices, vectors and arenas held
}

type buffer struct {
	buf, mem uint64
	size     uint64
	mapped   unsafe.Pointer
}

type pipeline struct {
	module, pipe uint64
}

// Open opens a device listed by Devices.
func Open(info Info) (*Device, error) {
	d := &Device{info: info}
	vk.GetPhysicalDeviceMemoryProperties(info.pd, &d.memory)

	var n uint32
	vk.GetPhysicalDeviceQueueFamilyProps(info.pd, &n, nil)
	fams := make([]byte, int(n)*24)
	if n > 0 {
		vk.GetPhysicalDeviceQueueFamilyProps(info.pd, &n, unsafe.Pointer(&fams[0]))
	}
	found := false
	for i := range int(n) {
		if binary.LittleEndian.Uint32(fams[i*24:])&queueCompute != 0 {
			d.family, found = uint32(i), true
			break
		}
	}
	if !found {
		return nil, errors.New("gpu: no compute queue")
	}
	prio := float32(1)
	qinfo := vkDeviceQueueCreateInfo{sType: stDeviceQueueCreateInfo, queueFamilyIndex: d.family, queueCount: 1, pQueuePriorities: &prio}
	dinfo := vkDeviceCreateInfo{sType: stDeviceCreateInfo, queueCreateInfoCount: 1, pQueueCreateInfos: &qinfo}
	var exts []*byte
	if slices.Contains(deviceExtensions(info.pd), "VK_KHR_portability_subset") {
		exts = append(exts, cstr("VK_KHR_portability_subset"))
	}
	if len(exts) > 0 {
		dinfo.enabledExtensionCount, dinfo.ppEnabledExtensionNames = uint32(len(exts)), &exts[0]
	}
	r := vk.CreateDevice(info.pd, &dinfo, nil, &d.dev)
	runtime.KeepAlive(exts)
	if err := check("create device", r); err != nil {
		return nil, err
	}
	vk.GetDeviceQueue(d.dev, d.family, 0, &d.queue)
	if err := d.init(); err != nil {
		d.Close()
		return nil, err
	}
	return d, nil
}

func (d *Device) init() error {
	pinfo := vkCommandPoolCreateInfo{sType: stCommandPoolCreate, flags: commandPoolResetBuffer, queueFamilyIndex: d.family}
	if err := check("create command pool", vk.CreateCommandPool(d.dev, &pinfo, nil, &d.pool)); err != nil {
		return err
	}
	ainfo := vkCommandBufferAllocateInfo{sType: stCommandBufferAllocate, commandPool: d.pool, commandBufferCount: 1}
	if err := check("allocate command buffer", vk.AllocateCommandBuffers(d.dev, &ainfo, &d.cmd)); err != nil {
		return err
	}
	finfo := vkFenceCreateInfo{sType: stFenceCreateInfo}
	if err := check("create fence", vk.CreateFence(d.dev, &finfo, nil, &d.fence)); err != nil {
		return err
	}
	for set, count := range []int{2, 1} {
		bs := make([]vkDescriptorSetLayoutBinding, count)
		for i := range bs {
			bs[i] = vkDescriptorSetLayoutBinding{binding: uint32(i), descriptorType: descriptorStorageBuffer, descriptorCount: 1, stageFlags: stageCompute}
		}
		linfo := vkDescriptorSetLayoutCreateInfo{sType: stDescriptorSetLayout, bindingCount: uint32(count), pBindings: &bs[0]}
		if err := check("create set layout", vk.CreateDescriptorSetLayout(d.dev, &linfo, nil, &d.setLayouts[set])); err != nil {
			return err
		}
	}
	pcr := vkPushConstantRange{stageFlags: stageCompute, size: 4 * pushWords}
	plinfo := vkPipelineLayoutCreateInfo{sType: stPipelineLayoutCreate, setLayoutCount: 2, pSetLayouts: &d.setLayouts[0],
		pushConstantRangeCount: 1, pPushConstantRanges: &pcr}
	if err := check("create pipeline layout", vk.CreatePipelineLayout(d.dev, &plinfo, nil, &d.layout)); err != nil {
		return err
	}
	// Arenas and layer vectors take one set each; a node holds at most a
	// few thousand layers and generations.
	size := vkDescriptorPoolSize{typ: descriptorStorageBuffer, count: 16384}
	poolInfo := vkDescriptorPoolCreateInfo{sType: stDescriptorPoolCreate, flags: descriptorPoolFreeSet, maxSets: 16384, poolSizeCount: 1, pPoolSizes: &size}
	if err := check("create descriptor pool", vk.CreateDescriptorPool(d.dev, &poolInfo, nil, &d.setPool)); err != nil {
		return err
	}
	for k := range kindCount {
		p, err := d.newPipeline(kernel(k, d.info.modes))
		if err != nil {
			return fmt.Errorf("kernel %d: %w", k, err)
		}
		d.kernels[k] = p
	}
	var err error
	d.host, err = d.NewArena(maxCols+maxRows+maxCols+maxCols/num.QBlock, maxCols+maxRows)
	if err != nil {
		return err
	}
	d.host.SetQuantScratch(maxCols+maxRows, maxCols+maxRows+maxCols)
	return nil
}

// Info describes the device.
func (d *Device) Info() Info { return d.info }

// Used is the device memory held by matrices, vectors and arenas, the
// device's own excepted.
func (d *Device) Used() int64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.used
}

// Close releases the device; what was allocated on it must be freed first.
func (d *Device) Close() {
	if d.dev == 0 {
		return
	}
	vk.QueueWaitIdle(d.queue)
	if d.host != nil {
		d.host.Free()
	}
	for _, p := range d.kernels {
		if p != nil {
			vk.DestroyPipeline(d.dev, p.pipe, nil)
			vk.DestroyShaderModule(d.dev, p.module, nil)
		}
	}
	if d.layout != 0 {
		vk.DestroyPipelineLayout(d.dev, d.layout, nil)
	}
	for _, l := range d.setLayouts {
		if l != 0 {
			vk.DestroyDescriptorSetLayout(d.dev, l, nil)
		}
	}
	if d.setPool != 0 {
		vk.DestroyDescriptorPool(d.dev, d.setPool, nil)
	}
	if d.fence != 0 {
		vk.DestroyFence(d.dev, d.fence, nil)
	}
	if d.pool != 0 {
		vk.DestroyCommandPool(d.dev, d.pool, nil)
	}
	vk.DestroyDevice(d.dev, nil)
	d.dev = 0
}

// memoryType finds a memory type allowed by bits with the flags wanted.
func (d *Device) memoryType(bits, want uint32) (uint32, bool) {
	count := binary.LittleEndian.Uint32(d.memory[0:])
	for i := range min(count, 32) {
		flags := binary.LittleEndian.Uint32(d.memory[4+8*i:])
		if bits&(1<<i) != 0 && flags&want == want {
			return i, true
		}
	}
	return 0, false
}

// newBuffer creates a storage buffer in memory with the flags wanted,
// mapped if host-visible.
func (d *Device) newBuffer(size uint64, want uint32) (*buffer, error) {
	b := &buffer{size: size}
	binfo := vkBufferCreateInfo{sType: stBufferCreateInfo, size: size, usage: bufferUsageStorage | bufferUsageTransferDst | bufferUsageTransferSrc}
	if err := check("create buffer", vk.CreateBuffer(d.dev, &binfo, nil, &b.buf)); err != nil {
		return nil, err
	}
	var req vkMemoryRequirements
	vk.GetBufferMemoryRequirements(d.dev, b.buf, &req)
	mt, ok := d.memoryType(req.memoryTypeBits, want)
	if !ok {
		vk.DestroyBuffer(d.dev, b.buf, nil)
		return nil, fmt.Errorf("gpu: no memory type with flags %#x", want)
	}
	minfo := vkMemoryAllocateInfo{sType: stMemoryAllocateInfo, allocationSize: req.size, memoryTypeIndex: mt}
	if err := check("allocate memory", vk.AllocateMemory(d.dev, &minfo, nil, &b.mem)); err != nil {
		vk.DestroyBuffer(d.dev, b.buf, nil)
		return nil, err
	}
	if err := check("bind memory", vk.BindBufferMemory(d.dev, b.buf, b.mem, 0)); err != nil {
		d.freeBuffer(b)
		return nil, err
	}
	flags := binary.LittleEndian.Uint32(d.memory[4+8*mt:])
	if flags&memHostVisible != 0 {
		if err := check("map memory", vk.MapMemory(d.dev, b.mem, 0, ^uint64(0), 0, &b.mapped)); err != nil {
			d.freeBuffer(b)
			return nil, err
		}
	}
	return b, nil
}

func (d *Device) freeBuffer(b *buffer) {
	if b == nil {
		return
	}
	if b.mapped != nil {
		vk.UnmapMemory(d.dev, b.mem)
	}
	vk.DestroyBuffer(d.dev, b.buf, nil)
	vk.FreeMemory(d.dev, b.mem, nil)
}

func (b *buffer) bytes() []byte { return unsafe.Slice((*byte)(b.mapped), b.size) }

// upload creates a device-local buffer holding data: written directly when
// the memory is also host-visible (integrated GPUs, resizable BAR), else
// through a staging buffer. The caller holds d.mu.
func (d *Device) upload(data []byte) (*buffer, error) {
	size := uint64(max(len(data), 4))
	b, err := d.newBuffer(size, memDeviceLocal)
	if err != nil {
		return nil, err
	}
	if b.mapped != nil {
		copy(b.bytes(), data)
		return b, nil
	}
	staging, err := d.newBuffer(size, memHostVisible|memHostCoherent)
	if err != nil {
		d.freeBuffer(b)
		return nil, err
	}
	defer d.freeBuffer(staging)
	copy(staging.bytes(), data)
	err = d.run(func(cb uintptr) {
		region := vkBufferCopy{size: size}
		vk.CmdCopyBuffer(cb, staging.buf, b.buf, 1, &region)
	})
	if err != nil {
		d.freeBuffer(b)
		return nil, err
	}
	return b, nil
}

// barrier makes every earlier write visible to every later command and to
// the host.
func barrier(cb uintptr) {
	bar := vkMemoryBarrier{sType: stMemoryBarrier, srcAccessMask: accessShaderWrite | accessTransferWrite,
		dstAccessMask: accessShaderRead | accessShaderWrite | accessTransferRead | accessTransferWrite | accessHostRead}
	stages := uint32(pipelineStageCompute | pipelineStageTransfer)
	vk.CmdPipelineBarrier(cb, stages, stages|pipelineStageHost, 0, 1, &bar, 0, nil, 0, nil)
}

// run records commands into the device's command buffer, submits them and
// waits. The caller holds d.mu.
func (d *Device) run(record func(cb uintptr)) error {
	begin := vkCommandBufferBeginInfo{sType: stCommandBufferBegin, flags: commandBufferOneTimeOnly}
	if err := check("begin commands", vk.BeginCommandBuffer(d.cmd, &begin)); err != nil {
		return err
	}
	record(d.cmd)
	barrier(d.cmd)
	return d.submit()
}

func (d *Device) submit() error {
	if err := check("end commands", vk.EndCommandBuffer(d.cmd)); err != nil {
		return err
	}
	cmd := d.cmd
	submit := vkSubmitInfo{sType: stSubmitInfo, commandBufferCount: 1, pCommandBuffers: &cmd}
	if err := check("submit", vk.QueueSubmit(d.queue, 1, &submit, d.fence)); err != nil {
		return err
	}
	r := vk.WaitForFences(d.dev, 1, &d.fence, 1, ^uint64(0))
	vk.ResetFences(d.dev, 1, &d.fence)
	return check("wait", r)
}

func (d *Device) newPipeline(code []uint32) (*pipeline, error) {
	p := &pipeline{}
	sinfo := vkShaderModuleCreateInfo{sType: stShaderModuleCreateInfo, codeSize: uintptr(4 * len(code)), pCode: &code[0]}
	if err := check("create shader", vk.CreateShaderModule(d.dev, &sinfo, nil, &p.module)); err != nil {
		return nil, err
	}
	cinfo := vkComputePipelineCreateInfo{sType: stComputePipelineCreate, layout: d.layout, basePipelineIndex: -1,
		stage: vkPipelineShaderStageCreateInfo{sType: stPipelineShaderStage, stage: stageCompute, module: p.module, pName: cstr("main")}}
	if err := check("create pipeline", vk.CreateComputePipelines(d.dev, 0, 1, &cinfo, nil, &p.pipe)); err != nil {
		return nil, err
	}
	return p, nil
}

// newSet allocates a descriptor set of layout set from pool, binding bufs.
func (d *Device) newSet(pool uint64, set int, bufs ...*buffer) (uint64, error) {
	layout := d.setLayouts[set]
	ainfo := vkDescriptorSetAllocateInfo{sType: stDescriptorSetAllocate, descriptorPool: pool, descriptorSetCount: 1, pSetLayouts: &layout}
	var ds uint64
	if err := check("allocate descriptor set", vk.AllocateDescriptorSets(d.dev, &ainfo, &ds)); err != nil {
		return 0, err
	}
	infos := make([]vkDescriptorBufferInfo, len(bufs))
	writes := make([]vkWriteDescriptorSet, len(bufs))
	for j, b := range bufs {
		infos[j] = vkDescriptorBufferInfo{buffer: b.buf, rng: b.size}
		writes[j] = vkWriteDescriptorSet{sType: stWriteDescriptorSet, dstSet: ds, dstBinding: uint32(j),
			descriptorCount: 1, descriptorType: descriptorStorageBuffer, pBufferInfo: &infos[j]}
	}
	vk.UpdateDescriptorSets(d.dev, uint32(len(writes)), &writes[0], 0, nil)
	runtime.KeepAlive(infos)
	return ds, nil
}

func floatBytes(v []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(v))), 4*len(v))
}

// Matrix is a weight matrix held on a device.
type Matrix struct {
	d          *Device
	kind       kernelKind
	rows, cols int
	// unrounded leaves FP8 and FP4 products in float32 (for tests).
	unrounded bool
	pool      uint64
	parts     []part
	bytes     int64
}

// part is a run of rows small enough for one storage buffer binding.
type part struct {
	row0, rows int
	w, s       *buffer // weights, and scales for quantized formats
	set        uint64
}

// Format is a matrix in one of the types package num stores.
type Format struct {
	F32    []float32
	BF16   []uint16
	Q8     []int8
	Q4     []byte
	Scales []float32
	// DeepSeek-V4.1's formats (package num): E4M3 codes with one E8M0
	// scale per 32×32 block ([ceil(rows/32)][cols/32]), or E2M1 codes,
	// two to a byte, with one per 32 of a row ([rows][cols/32]); the
	// scales' bytes in E8M0. The products are rounded to bfloat16.
	FP8, FP4 []byte
	E8M0     []byte
	// FP4Chunks and E8M0Chunks are FP4 and E8M0 given in pieces, read as
	// one: stacked experts, which are uploaded part by part and so never
	// gathered on the host whole.
	FP4Chunks, E8M0Chunks [][]byte
	// AlignRows, if set, keeps runs of that many rows (an expert's) in one
	// part when the matrix is split.
	AlignRows int
}

// Upload copies a rows×cols matrix to the device.
func (d *Device) Upload(rows, cols int, f Format) (*Matrix, error) {
	m := &Matrix{d: d, rows: rows, cols: cols}
	var rowBytes, scaleBytes int
	var data []byte
	switch {
	case f.F32 != nil:
		m.kind, rowBytes = kindF32, 4*cols
		data = floatBytes(f.F32)
	case f.BF16 != nil:
		if cols%2 != 0 {
			return nil, errors.New("gpu: bfloat16 rows need an even width")
		}
		m.kind, rowBytes = kindBF16, 2*cols
		data = unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(f.BF16))), 2*len(f.BF16))
	case f.Q8 != nil:
		m.kind, rowBytes, scaleBytes = kindQ8, cols, 4*cols/num.QBlock
		data = unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(f.Q8))), len(f.Q8))
	case f.Q4 != nil:
		m.kind, rowBytes, scaleBytes = kindQ4, cols/2, 4*cols/num.QBlock
		data = f.Q4
	case f.FP8 != nil:
		m.kind, rowBytes = kindFP8, cols
		data = f.FP8
	case f.FP4 != nil:
		m.kind, rowBytes = kindFP4, cols/2
		data = f.FP4
	case f.FP4Chunks != nil:
		m.kind, rowBytes = kindFP4, cols/2
	default:
		return nil, errors.New("gpu: empty matrix")
	}
	// Rows beyond maxRows are for products into an arena (stacked experts);
	// MulVecs, through the host's arena, checks its own outputs.
	read, dataLen := gatherer(data, f.FP4Chunks)
	if cols > maxCols || rows > 1<<24 || dataLen != rows*rowBytes {
		return nil, fmt.Errorf("gpu: matrix %d×%d out of range", rows, cols)
	}
	if (m.kind == kindQ8 || m.kind == kindQ4) && (cols%num.QBlock != 0 || len(f.Scales) != rows*cols/num.QBlock) {
		return nil, errors.New("gpu: quantized matrix of the wrong shape")
	}
	scales := floatBytes(f.Scales)
	mx := m.kind == kindFP8 || m.kind == kindFP4
	if mx {
		blocks := rows * cols / num.QBlock // one scale per 32 of a row
		if m.kind == kindFP8 {
			blocks = (rows + 31) / 32 * (cols / num.QBlock)
		}
		if _, n := gatherer(f.E8M0, f.E8M0Chunks); cols%num.QBlock != 0 || n != blocks {
			return nil, errors.New("gpu: block-scaled matrix of the wrong shape")
		}
		if m.kind == kindFP8 && !num.ValidE4M3(f.FP8) {
			return nil, errors.New("gpu: NaN in E4M3 weights")
		}
		scales = f.E8M0
	}
	readScales, _ := gatherer(scales, f.E8M0Chunks)

	// Split rows so each part fits a binding, and a gigabyte at most: for
	// E4M3 weights, whole blocks of 32 rows sharing scales.
	limit := min(d.info.maxRange, 1<<30)
	per := max(int(limit)/rowBytes, 1)
	if m.kind == kindFP8 {
		per = max(per/32*32, 32)
	}
	if a := f.AlignRows; a > 0 {
		if rows%a != 0 || m.kind == kindFP8 && a%32 != 0 {
			return nil, errors.New("gpu: rows not a whole number of aligned runs")
		}
		per = max(per/a*a, a)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	size := vkDescriptorPoolSize{typ: descriptorStorageBuffer, count: uint32(2 * ((rows + per - 1) / per))}
	pinfo := vkDescriptorPoolCreateInfo{sType: stDescriptorPoolCreate, maxSets: size.count / 2, poolSizeCount: 1, pPoolSizes: &size}
	if err := check("create descriptor pool", vk.CreateDescriptorPool(d.dev, &pinfo, nil, &m.pool)); err != nil {
		return nil, err
	}
	for r := 0; r < rows; r += per {
		n := min(per, rows-r)
		p := part{row0: r, rows: n}
		var err error
		if p.w, err = d.upload(read(r*rowBytes, (r+n)*rowBytes)); err == nil {
			// Byte scales are padded to whole words, which kernels read.
			nb := cols / num.QBlock
			switch {
			case m.kind == kindFP8:
				p.s, err = d.upload(padWords(readScales(r/32*nb, (r+n+31)/32*nb)))
			case m.kind == kindFP4:
				p.s, err = d.upload(padWords(readScales(r*nb, (r+n)*nb)))
			case scaleBytes > 0:
				p.s, err = d.upload(scales[r*scaleBytes : (r+n)*scaleBytes])
			}
		}
		if err == nil {
			bufs := []*buffer{p.w}
			if p.s != nil {
				bufs = append(bufs, p.s)
			}
			p.set, err = d.newSet(m.pool, 0, bufs...)
		}
		if err != nil {
			d.freeBuffer(p.w)
			d.freeBuffer(p.s)
			m.free()
			return nil, err
		}
		m.parts = append(m.parts, p)
		m.bytes += int64(p.w.size)
		if p.s != nil {
			m.bytes += int64(p.s.size)
		}
	}
	d.used += m.bytes
	runtime.KeepAlive(f)
	return m, nil
}

// gatherer reads bytes [lo, hi) of data, or of chunks read as one (only
// that range is gathered), and reports their whole length.
func gatherer(data []byte, chunks [][]byte) (func(lo, hi int) []byte, int) {
	if chunks == nil {
		return func(lo, hi int) []byte { return data[lo:hi] }, len(data)
	}
	n := 0
	for _, c := range chunks {
		n += len(c)
	}
	return func(lo, hi int) []byte {
		out := make([]byte, 0, hi-lo)
		at := 0
		for _, c := range chunks {
			if end := at + len(c); end > lo && at < hi {
				out = append(out, c[max(lo-at, 0):min(hi-at, len(c))]...)
			}
			at += len(c)
		}
		return out
	}, n
}

// padWords pads bytes to whole words.
func padWords(b []byte) []byte {
	if len(b)%4 == 0 {
		return b
	}
	return append(append([]byte(nil), b...), make([]byte, 4-len(b)%4)...)
}

// SameDevice reports whether two matrices are on the same device.
func SameDevice(a, b *Matrix) bool { return a != nil && b != nil && a.d == b.d }

// Bytes is the device memory the matrix holds.
func (m *Matrix) Bytes() int64 { return m.bytes }

// Device is the device holding the matrix.
func (m *Matrix) Device() *Device { return m.d }

// Free releases the matrix's device memory.
func (m *Matrix) Free() {
	m.d.mu.Lock()
	defer m.d.mu.Unlock()
	m.free()
	m.d.used -= m.bytes
}

func (m *Matrix) free() {
	for _, p := range m.parts {
		m.d.freeBuffer(p.w)
		m.d.freeBuffer(p.s)
	}
	if m.pool != 0 {
		vk.DestroyDescriptorPool(m.d.dev, m.pool, nil)
	}
	m.parts, m.pool = nil, 0
}

// MulVec sets out = m·x, as package num would compute it.
func (m *Matrix) MulVec(out, x []float32) error {
	return MulVecs([]*Matrix{m}, [][]float32{out}, x)
}

// MulVecs computes several products of matrices on one device with the
// same input in a single submission, which saves the per-submission cost
// for, say, a layer's query, key and value projections.
func MulVecs(ms []*Matrix, outs [][]float32, x []float32) error {
	if len(ms) == 0 {
		return nil
	}
	d := ms[0].d
	total := 0
	offs := make([]int, len(ms))
	for i, m := range ms {
		if m.d != d || len(x) != m.cols || len(outs[i]) != m.rows {
			return fmt.Errorf("gpu: %d×%d matrix times %d into %d", m.rows, m.cols, len(x), len(outs[i]))
		}
		offs[i] = maxCols + total
		total += m.rows
	}
	if total > maxRows {
		return errors.New("gpu: products too large for one submission")
	}
	b := d.host.Begin()
	b.Write(0, x)
	b.MatVecs(ms, offs, 0)
	for i := range ms {
		b.Read(offs[i], outs[i])
	}
	return b.Submit()
}
