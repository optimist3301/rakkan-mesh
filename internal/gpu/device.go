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

// I/O buffer sizes: the widest input and output a model may have.
const (
	maxCols = 1 << 18 // Config.Hidden's limit
	maxRows = 1 << 20 // Config.Vocab's limit
)

// Device is an opened GPU. Its methods are safe for concurrent use; the
// products run one at a time.
type Device struct {
	info   Info
	dev    uintptr
	queue  uintptr
	family uint32
	memory [520]byte

	pool  uint64
	cmd   uintptr
	fence uint64

	kernels map[kernelKind]*pipeline
	in, out *buffer // input vector (or quantized input and scales), output
	qx, xs  *buffer

	mu   sync.Mutex
	used int64 // bytes of matrices held
}

type buffer struct {
	buf, mem uint64
	size     uint64
	mapped   unsafe.Pointer
}

type pipeline struct {
	module, setLayout, layout, pipe uint64
	bindings                        int
}

// Open opens a device listed by Devices.
func Open(info Info) (*Device, error) {
	d := &Device{info: info, kernels: map[kernelKind]*pipeline{}}
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
	var err error
	host := uint32(memHostVisible | memHostCoherent)
	if d.in, err = d.newBuffer(4*maxCols, host); err != nil {
		return err
	}
	if d.out, err = d.newBuffer(4*maxRows, host); err != nil {
		return err
	}
	if d.qx, err = d.newBuffer(maxCols, host); err != nil {
		return err
	}
	if d.xs, err = d.newBuffer(4*maxCols/num.QBlock, host); err != nil {
		return err
	}
	for _, k := range []kernelKind{kindF32, kindBF16} {
		if d.kernels[k], err = d.newPipeline(floatKernel(k, d.info.modes), 3); err != nil {
			return err
		}
	}
	for _, k := range []kernelKind{kindQ8, kindQ4} {
		if d.kernels[k], err = d.newPipeline(quantKernel(k, d.info.modes), 5); err != nil {
			return err
		}
	}
	return nil
}

// Info describes the device.
func (d *Device) Info() Info { return d.info }

// Used is the memory held by matrices on the device.
func (d *Device) Used() int64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.used
}

// Close releases the device; its matrices must be freed first.
func (d *Device) Close() {
	if d.dev == 0 {
		return
	}
	vk.QueueWaitIdle(d.queue)
	for _, p := range d.kernels {
		vk.DestroyPipeline(d.dev, p.pipe, nil)
		vk.DestroyPipelineLayout(d.dev, p.layout, nil)
		vk.DestroyDescriptorSetLayout(d.dev, p.setLayout, nil)
		vk.DestroyShaderModule(d.dev, p.module, nil)
	}
	for _, b := range []*buffer{d.in, d.out, d.qx, d.xs} {
		d.freeBuffer(b)
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
// through a staging buffer.
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

// run records commands into the device's command buffer, submits them and
// waits. The caller holds d.mu or has the device to itself.
func (d *Device) run(record func(cb uintptr)) error {
	begin := vkCommandBufferBeginInfo{sType: stCommandBufferBegin, flags: commandBufferOneTimeOnly}
	if err := check("begin commands", vk.BeginCommandBuffer(d.cmd, &begin)); err != nil {
		return err
	}
	record(d.cmd)
	// Results become visible to the host.
	bar := vkMemoryBarrier{sType: stMemoryBarrier, srcAccessMask: accessShaderWrite | accessTransferWrite, dstAccessMask: accessHostRead}
	vk.CmdPipelineBarrier(d.cmd, pipelineStageCompute|pipelineStageTransfer, pipelineStageHost, 0, 1, &bar, 0, nil, 0, nil)
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

func (d *Device) newPipeline(code []uint32, bindings int) (*pipeline, error) {
	p := &pipeline{bindings: bindings}
	sinfo := vkShaderModuleCreateInfo{sType: stShaderModuleCreateInfo, codeSize: uintptr(4 * len(code)), pCode: &code[0]}
	if err := check("create shader", vk.CreateShaderModule(d.dev, &sinfo, nil, &p.module)); err != nil {
		return nil, err
	}
	bs := make([]vkDescriptorSetLayoutBinding, bindings)
	for i := range bs {
		bs[i] = vkDescriptorSetLayoutBinding{binding: uint32(i), descriptorType: descriptorStorageBuffer, descriptorCount: 1, stageFlags: stageCompute}
	}
	linfo := vkDescriptorSetLayoutCreateInfo{sType: stDescriptorSetLayout, bindingCount: uint32(bindings), pBindings: &bs[0]}
	if err := check("create set layout", vk.CreateDescriptorSetLayout(d.dev, &linfo, nil, &p.setLayout)); err != nil {
		return nil, err
	}
	pcr := vkPushConstantRange{stageFlags: stageCompute, size: 16}
	plinfo := vkPipelineLayoutCreateInfo{sType: stPipelineLayoutCreate, setLayoutCount: 1, pSetLayouts: &p.setLayout,
		pushConstantRangeCount: 1, pPushConstantRanges: &pcr}
	if err := check("create pipeline layout", vk.CreatePipelineLayout(d.dev, &plinfo, nil, &p.layout)); err != nil {
		return nil, err
	}
	cinfo := vkComputePipelineCreateInfo{sType: stComputePipelineCreate, layout: p.layout, basePipelineIndex: -1,
		stage: vkPipelineShaderStageCreateInfo{sType: stPipelineShaderStage, stage: stageCompute, module: p.module, pName: cstr("main")}}
	if err := check("create pipeline", vk.CreateComputePipelines(d.dev, 0, 1, &cinfo, nil, &p.pipe)); err != nil {
		return nil, err
	}
	return p, nil
}

// Matrix is a weight matrix held on a device.
type Matrix struct {
	d          *Device
	kind       kernelKind
	rows, cols int
	pool       uint64
	parts      []part
	bytes      int64
}

// part is a run of rows small enough for one storage buffer binding.
type part struct {
	row0, rows int
	w, s       *buffer // weights, and scales for quantized formats
	set        uint64
}

// Matrices come in these formats, as package num stores them.
type Format struct {
	F32    []float32
	BF16   []uint16
	Q8     []int8
	Q4     []byte
	Scales []float32
}

// Upload copies a rows×cols matrix to the device.
func (d *Device) Upload(rows, cols int, f Format) (*Matrix, error) {
	m := &Matrix{d: d, rows: rows, cols: cols}
	var rowBytes, scaleBytes int
	var data []byte
	switch {
	case f.F32 != nil:
		m.kind, rowBytes = kindF32, 4*cols
		data = unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(f.F32))), 4*len(f.F32))
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
	default:
		return nil, errors.New("gpu: empty matrix")
	}
	if cols > maxCols || rows > maxRows || len(data) != rows*rowBytes {
		return nil, fmt.Errorf("gpu: matrix %d×%d out of range", rows, cols)
	}
	if (m.kind == kindQ8 || m.kind == kindQ4) && (cols%num.QBlock != 0 || len(f.Scales) != rows*cols/num.QBlock) {
		return nil, errors.New("gpu: quantized matrix of the wrong shape")
	}
	scales := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(f.Scales))), 4*len(f.Scales))

	// Split rows so each part fits a binding, and a gigabyte at most.
	limit := min(d.info.maxRange, 1<<30)
	per := max(int(limit)/rowBytes, 1)
	d.mu.Lock()
	defer d.mu.Unlock()
	for r := 0; r < rows; r += per {
		n := min(per, rows-r)
		p := part{row0: r, rows: n}
		var err error
		if p.w, err = d.upload(data[r*rowBytes : (r+n)*rowBytes]); err == nil && scaleBytes > 0 {
			p.s, err = d.upload(scales[r*scaleBytes : (r+n)*scaleBytes])
		}
		if err != nil {
			m.free()
			return nil, err
		}
		m.parts = append(m.parts, p)
		m.bytes += int64(p.w.size)
		if p.s != nil {
			m.bytes += int64(p.s.size)
		}
	}
	if err := m.bind(); err != nil {
		m.free()
		return nil, err
	}
	d.used += m.bytes
	runtime.KeepAlive(f)
	return m, nil
}

// bind creates a descriptor set per part.
func (m *Matrix) bind() error {
	d := m.d
	p := d.kernels[m.kind]
	size := vkDescriptorPoolSize{typ: descriptorStorageBuffer, count: uint32(len(m.parts) * p.bindings)}
	pinfo := vkDescriptorPoolCreateInfo{sType: stDescriptorPoolCreate, maxSets: uint32(len(m.parts)), poolSizeCount: 1, pPoolSizes: &size}
	if err := check("create descriptor pool", vk.CreateDescriptorPool(d.dev, &pinfo, nil, &m.pool)); err != nil {
		return err
	}
	for i := range m.parts {
		pt := &m.parts[i]
		layout := p.setLayout
		ainfo := vkDescriptorSetAllocateInfo{sType: stDescriptorSetAllocate, descriptorPool: m.pool, descriptorSetCount: 1, pSetLayouts: &layout}
		if err := check("allocate descriptor set", vk.AllocateDescriptorSets(d.dev, &ainfo, &pt.set)); err != nil {
			return err
		}
		bufs := []*buffer{pt.w, d.in, d.out}
		if m.kind == kindQ8 || m.kind == kindQ4 {
			bufs = []*buffer{pt.w, pt.s, d.qx, d.xs, d.out}
		}
		infos := make([]vkDescriptorBufferInfo, len(bufs))
		writes := make([]vkWriteDescriptorSet, len(bufs))
		for j, b := range bufs {
			infos[j] = vkDescriptorBufferInfo{buffer: b.buf, rng: b.size}
			writes[j] = vkWriteDescriptorSet{sType: stWriteDescriptorSet, dstSet: pt.set, dstBinding: uint32(j),
				descriptorCount: 1, descriptorType: descriptorStorageBuffer, pBufferInfo: &infos[j]}
		}
		vk.UpdateDescriptorSets(d.dev, uint32(len(writes)), &writes[0], 0, nil)
		runtime.KeepAlive(infos)
	}
	return nil
}

// SameDevice reports whether two matrices are on the same device.
func SameDevice(a, b *Matrix) bool { return a != nil && b != nil && a.d == b.d }

// Bytes is the device memory the matrix holds.
func (m *Matrix) Bytes() int64 { return m.bytes }

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
	total, quant, float := 0, false, false
	for i, m := range ms {
		if m.d != d || len(x) != m.cols || len(outs[i]) != m.rows {
			return fmt.Errorf("gpu: %d×%d matrix times %d into %d", m.rows, m.cols, len(x), len(outs[i]))
		}
		total += m.rows
		quant = quant || m.kind == kindQ8 || m.kind == kindQ4
		float = float || m.kind == kindF32 || m.kind == kindBF16
	}
	if total > maxRows {
		return errors.New("gpu: products too large for one submission")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if quant {
		qx, xs := num.QuantizeActs(x)
		copy(d.qx.bytes(), unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(qx))), len(qx)))
		copy(d.xs.bytes(), unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(xs))), 4*len(xs)))
	}
	if float {
		copy(d.in.bytes(), unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(x))), 4*len(x)))
	}
	err := d.run(func(cb uintptr) {
		at := 0
		for _, m := range ms {
			p := d.kernels[m.kind]
			vk.CmdBindPipeline(cb, bindPointCompute, p.pipe)
			for _, pt := range m.parts {
				set := pt.set
				vk.CmdBindDescriptorSets(cb, bindPointCompute, p.layout, 0, 1, &set, 0, nil)
				per := groupRows(m.kind)
				groups := (pt.rows + per - 1) / per
				gx := min(groups, 65535)
				gy := (groups + gx - 1) / gx
				pc := [4]uint32{uint32(pt.rows), uint32(m.cols), uint32(gx), uint32(at + pt.row0)}
				vk.CmdPushConstants(cb, p.layout, stageCompute, 0, 16, unsafe.Pointer(&pc[0]))
				vk.CmdDispatch(cb, uint32(gx), uint32(gy), 1)
			}
			at += m.rows
		}
	})
	if err != nil {
		return err
	}
	res := unsafe.Slice((*float32)(d.out.mapped), total)
	for i, m := range ms {
		copy(outs[i], res[:m.rows])
		res = res[m.rows:]
	}
	return nil
}
