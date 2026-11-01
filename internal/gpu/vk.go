package gpu

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

// Vulkan, called through purego: no cgo, so the node still builds as a
// static binary, and a machine without a Vulkan loader simply has no GPU.
// The structures below mirror the C layouts of 64-bit platforms, where Go
// aligns fields as C does.

const (
	vkSuccess = 0
	vkTimeout = 2

	stApplicationInfo         = 0
	stInstanceCreateInfo      = 1
	stDeviceQueueCreateInfo   = 2
	stDeviceCreateInfo        = 3
	stSubmitInfo              = 4
	stMemoryAllocateInfo      = 5
	stFenceCreateInfo         = 8
	stBufferCreateInfo        = 12
	stShaderModuleCreateInfo  = 16
	stPipelineShaderStage     = 18
	stComputePipelineCreate   = 29
	stPipelineLayoutCreate    = 30
	stDescriptorSetLayout     = 32
	stDescriptorPoolCreate    = 33
	stDescriptorSetAllocate   = 34
	stWriteDescriptorSet      = 35
	stCommandPoolCreate       = 39
	stCommandBufferAllocate   = 40
	stCommandBufferBegin      = 42
	stMemoryBarrier           = 46
	stPhysicalDeviceProps2    = 1000059001
	stFloatControlsProperties = 1000197000

	bufferUsageTransferSrc = 0x1
	bufferUsageTransferDst = 0x2
	bufferUsageStorage     = 0x20

	memDeviceLocal  = 0x1
	memHostVisible  = 0x2
	memHostCoherent = 0x4

	queueCompute = 0x2

	descriptorStorageBuffer = 7
	stageCompute            = 0x20
	bindPointCompute        = 1

	pipelineStageTransfer = 0x1000
	pipelineStageCompute  = 0x800
	pipelineStageHost     = 0x4000
	accessShaderRead      = 0x20
	accessShaderWrite     = 0x40
	accessTransferWrite   = 0x1000
	accessHostRead        = 0x2000
	accessHostWrite       = 0x4000

	commandPoolResetBuffer   = 0x2
	commandBufferOneTimeOnly = 0x1

	physicalDeviceOther      = 0
	physicalDeviceIntegrated = 1
	physicalDeviceDiscrete   = 2
	physicalDeviceVirtual    = 3
	physicalDeviceCPU        = 4

	instanceEnumeratePortability = 0x1
)

func apiVersion(major, minor uint32) uint32 { return major<<22 | minor<<12 }

type vkApplicationInfo struct {
	sType              uint32
	pNext              unsafe.Pointer
	pApplicationName   *byte
	applicationVersion uint32
	pEngineName        *byte
	engineVersion      uint32
	apiVersion         uint32
}

type vkInstanceCreateInfo struct {
	sType                   uint32
	pNext                   unsafe.Pointer
	flags                   uint32
	pApplicationInfo        *vkApplicationInfo
	enabledLayerCount       uint32
	ppEnabledLayerNames     **byte
	enabledExtensionCount   uint32
	ppEnabledExtensionNames **byte
}

type vkDeviceQueueCreateInfo struct {
	sType            uint32
	pNext            unsafe.Pointer
	flags            uint32
	queueFamilyIndex uint32
	queueCount       uint32
	pQueuePriorities *float32
}

type vkDeviceCreateInfo struct {
	sType                   uint32
	pNext                   unsafe.Pointer
	flags                   uint32
	queueCreateInfoCount    uint32
	pQueueCreateInfos       *vkDeviceQueueCreateInfo
	enabledLayerCount       uint32
	ppEnabledLayerNames     **byte
	enabledExtensionCount   uint32
	ppEnabledExtensionNames **byte
	pEnabledFeatures        unsafe.Pointer
}

type vkProperties2 struct {
	sType uint32
	pNext unsafe.Pointer
	props [1024]byte // VkPhysicalDeviceProperties, 824 bytes, and slack
}

type vkFloatControls struct {
	sType                      uint32
	pNext                      unsafe.Pointer
	denormBehaviorIndependence uint32
	roundingModeIndependence   uint32
	// SignedZeroInfNanPreserve, DenormPreserve, DenormFlushToZero,
	// RoundingModeRTE, RoundingModeRTZ; each for float16, 32 and 64.
	flags [15]uint32
}

type vkBufferCreateInfo struct {
	sType                 uint32
	pNext                 unsafe.Pointer
	flags                 uint32
	size                  uint64
	usage                 uint32
	sharingMode           uint32
	queueFamilyIndexCount uint32
	pQueueFamilyIndices   *uint32
}

type vkMemoryRequirements struct {
	size           uint64
	alignment      uint64
	memoryTypeBits uint32
}

type vkMemoryAllocateInfo struct {
	sType           uint32
	pNext           unsafe.Pointer
	allocationSize  uint64
	memoryTypeIndex uint32
}

type vkShaderModuleCreateInfo struct {
	sType    uint32
	pNext    unsafe.Pointer
	flags    uint32
	codeSize uintptr
	pCode    *uint32
}

type vkDescriptorSetLayoutBinding struct {
	binding            uint32
	descriptorType     uint32
	descriptorCount    uint32
	stageFlags         uint32
	pImmutableSamplers unsafe.Pointer
}

type vkDescriptorSetLayoutCreateInfo struct {
	sType        uint32
	pNext        unsafe.Pointer
	flags        uint32
	bindingCount uint32
	pBindings    *vkDescriptorSetLayoutBinding
}

type vkPushConstantRange struct {
	stageFlags uint32
	offset     uint32
	size       uint32
}

type vkPipelineLayoutCreateInfo struct {
	sType                  uint32
	pNext                  unsafe.Pointer
	flags                  uint32
	setLayoutCount         uint32
	pSetLayouts            *uint64
	pushConstantRangeCount uint32
	pPushConstantRanges    *vkPushConstantRange
}

type vkPipelineShaderStageCreateInfo struct {
	sType               uint32
	pNext               unsafe.Pointer
	flags               uint32
	stage               uint32
	module              uint64
	pName               *byte
	pSpecializationInfo unsafe.Pointer
}

type vkComputePipelineCreateInfo struct {
	sType              uint32
	pNext              unsafe.Pointer
	flags              uint32
	stage              vkPipelineShaderStageCreateInfo
	layout             uint64
	basePipelineHandle uint64
	basePipelineIndex  int32
}

type vkDescriptorPoolSize struct {
	typ   uint32
	count uint32
}

type vkDescriptorPoolCreateInfo struct {
	sType         uint32
	pNext         unsafe.Pointer
	flags         uint32
	maxSets       uint32
	poolSizeCount uint32
	pPoolSizes    *vkDescriptorPoolSize
}

type vkDescriptorSetAllocateInfo struct {
	sType              uint32
	pNext              unsafe.Pointer
	descriptorPool     uint64
	descriptorSetCount uint32
	pSetLayouts        *uint64
}

type vkDescriptorBufferInfo struct {
	buffer uint64
	offset uint64
	rng    uint64
}

type vkWriteDescriptorSet struct {
	sType            uint32
	pNext            unsafe.Pointer
	dstSet           uint64
	dstBinding       uint32
	dstArrayElement  uint32
	descriptorCount  uint32
	descriptorType   uint32
	pImageInfo       unsafe.Pointer
	pBufferInfo      *vkDescriptorBufferInfo
	pTexelBufferView unsafe.Pointer
}

type vkCommandPoolCreateInfo struct {
	sType            uint32
	pNext            unsafe.Pointer
	flags            uint32
	queueFamilyIndex uint32
}

type vkCommandBufferAllocateInfo struct {
	sType              uint32
	pNext              unsafe.Pointer
	commandPool        uint64
	level              uint32
	commandBufferCount uint32
}

type vkCommandBufferBeginInfo struct {
	sType            uint32
	pNext            unsafe.Pointer
	flags            uint32
	pInheritanceInfo unsafe.Pointer
}

type vkMemoryBarrier struct {
	sType         uint32
	pNext         unsafe.Pointer
	srcAccessMask uint32
	dstAccessMask uint32
}

type vkBufferCopy struct {
	srcOffset uint64
	dstOffset uint64
	size      uint64
}

type vkSubmitInfo struct {
	sType                uint32
	pNext                unsafe.Pointer
	waitSemaphoreCount   uint32
	pWaitSemaphores      unsafe.Pointer
	pWaitDstStageMask    unsafe.Pointer
	commandBufferCount   uint32
	pCommandBuffers      *uintptr
	signalSemaphoreCount uint32
	pSignalSemaphores    unsafe.Pointer
}

type vkFenceCreateInfo struct {
	sType uint32
	pNext unsafe.Pointer
	flags uint32
}

// vk holds the Vulkan entry points.
var vk struct {
	EnumerateInstanceVersion             func(*uint32) int32
	CreateInstance                       func(*vkInstanceCreateInfo, unsafe.Pointer, *uintptr) int32
	DestroyInstance                      func(uintptr, unsafe.Pointer)
	EnumerateInstanceExtensionProperties func(*byte, *uint32, unsafe.Pointer) int32
	EnumeratePhysicalDevices             func(uintptr, *uint32, *uintptr) int32
	GetPhysicalDeviceProperties2         func(uintptr, *vkProperties2)
	GetPhysicalDeviceMemoryProperties    func(uintptr, *[520]byte)
	GetPhysicalDeviceQueueFamilyProps    func(uintptr, *uint32, unsafe.Pointer)
	EnumerateDeviceExtensionProperties   func(uintptr, *byte, *uint32, unsafe.Pointer) int32
	CreateDevice                         func(uintptr, *vkDeviceCreateInfo, unsafe.Pointer, *uintptr) int32
	DestroyDevice                        func(uintptr, unsafe.Pointer)
	GetDeviceQueue                       func(uintptr, uint32, uint32, *uintptr)
	CreateBuffer                         func(uintptr, *vkBufferCreateInfo, unsafe.Pointer, *uint64) int32
	DestroyBuffer                        func(uintptr, uint64, unsafe.Pointer)
	GetBufferMemoryRequirements          func(uintptr, uint64, *vkMemoryRequirements)
	AllocateMemory                       func(uintptr, *vkMemoryAllocateInfo, unsafe.Pointer, *uint64) int32
	FreeMemory                           func(uintptr, uint64, unsafe.Pointer)
	BindBufferMemory                     func(uintptr, uint64, uint64, uint64) int32
	MapMemory                            func(uintptr, uint64, uint64, uint64, uint32, *unsafe.Pointer) int32
	UnmapMemory                          func(uintptr, uint64)
	CreateShaderModule                   func(uintptr, *vkShaderModuleCreateInfo, unsafe.Pointer, *uint64) int32
	DestroyShaderModule                  func(uintptr, uint64, unsafe.Pointer)
	CreateDescriptorSetLayout            func(uintptr, *vkDescriptorSetLayoutCreateInfo, unsafe.Pointer, *uint64) int32
	DestroyDescriptorSetLayout           func(uintptr, uint64, unsafe.Pointer)
	CreatePipelineLayout                 func(uintptr, *vkPipelineLayoutCreateInfo, unsafe.Pointer, *uint64) int32
	DestroyPipelineLayout                func(uintptr, uint64, unsafe.Pointer)
	CreateComputePipelines               func(uintptr, uint64, uint32, *vkComputePipelineCreateInfo, unsafe.Pointer, *uint64) int32
	DestroyPipeline                      func(uintptr, uint64, unsafe.Pointer)
	CreateDescriptorPool                 func(uintptr, *vkDescriptorPoolCreateInfo, unsafe.Pointer, *uint64) int32
	DestroyDescriptorPool                func(uintptr, uint64, unsafe.Pointer)
	AllocateDescriptorSets               func(uintptr, *vkDescriptorSetAllocateInfo, *uint64) int32
	UpdateDescriptorSets                 func(uintptr, uint32, *vkWriteDescriptorSet, uint32, unsafe.Pointer)
	CreateCommandPool                    func(uintptr, *vkCommandPoolCreateInfo, unsafe.Pointer, *uint64) int32
	DestroyCommandPool                   func(uintptr, uint64, unsafe.Pointer)
	AllocateCommandBuffers               func(uintptr, *vkCommandBufferAllocateInfo, *uintptr) int32
	FreeCommandBuffers                   func(uintptr, uint64, uint32, *uintptr)
	BeginCommandBuffer                   func(uintptr, *vkCommandBufferBeginInfo) int32
	EndCommandBuffer                     func(uintptr) int32
	CmdBindPipeline                      func(uintptr, uint32, uint64)
	CmdBindDescriptorSets                func(uintptr, uint32, uint64, uint32, uint32, *uint64, uint32, unsafe.Pointer)
	CmdPushConstants                     func(uintptr, uint64, uint32, uint32, uint32, unsafe.Pointer)
	CmdDispatch                          func(uintptr, uint32, uint32, uint32)
	CmdPipelineBarrier                   func(uintptr, uint32, uint32, uint32, uint32, *vkMemoryBarrier, uint32, unsafe.Pointer, uint32, unsafe.Pointer)
	CmdCopyBuffer                        func(uintptr, uint64, uint64, uint32, *vkBufferCopy)
	QueueSubmit                          func(uintptr, uint32, *vkSubmitInfo, uint64) int32
	QueueWaitIdle                        func(uintptr) int32
	CreateFence                          func(uintptr, *vkFenceCreateInfo, unsafe.Pointer, *uint64) int32
	DestroyFence                         func(uintptr, uint64, unsafe.Pointer)
	WaitForFences                        func(uintptr, uint32, *uint64, uint32, uint64) int32
	ResetFences                          func(uintptr, uint32, *uint64) int32
}

var (
	loadOnce sync.Once
	loadErr  error
)

// ErrNoVulkan means the machine has no usable Vulkan loader.
var ErrNoVulkan = errors.New("gpu: no Vulkan loader")

// load opens the Vulkan loader and binds every entry point used.
func load() error {
	loadOnce.Do(func() {
		if unsafe.Sizeof(uintptr(0)) != 8 {
			loadErr = errors.New("gpu: 64-bit platforms only")
			return
		}
		lib, err := openLibrary()
		if err != nil {
			loadErr = fmt.Errorf("%w: %v", ErrNoVulkan, err)
			return
		}
		bind := func(fn any, name string) {
			if loadErr != nil {
				return
			}
			defer func() {
				if r := recover(); r != nil {
					loadErr = fmt.Errorf("%w: %s: %v", ErrNoVulkan, name, r)
				}
			}()
			registerFunc(fn, lib, name)
		}
		bind(&vk.EnumerateInstanceVersion, "vkEnumerateInstanceVersion")
		bind(&vk.CreateInstance, "vkCreateInstance")
		bind(&vk.DestroyInstance, "vkDestroyInstance")
		bind(&vk.EnumerateInstanceExtensionProperties, "vkEnumerateInstanceExtensionProperties")
		bind(&vk.EnumeratePhysicalDevices, "vkEnumeratePhysicalDevices")
		bind(&vk.GetPhysicalDeviceProperties2, "vkGetPhysicalDeviceProperties2")
		bind(&vk.GetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties")
		bind(&vk.GetPhysicalDeviceQueueFamilyProps, "vkGetPhysicalDeviceQueueFamilyProperties")
		bind(&vk.EnumerateDeviceExtensionProperties, "vkEnumerateDeviceExtensionProperties")
		bind(&vk.CreateDevice, "vkCreateDevice")
		bind(&vk.DestroyDevice, "vkDestroyDevice")
		bind(&vk.GetDeviceQueue, "vkGetDeviceQueue")
		bind(&vk.CreateBuffer, "vkCreateBuffer")
		bind(&vk.DestroyBuffer, "vkDestroyBuffer")
		bind(&vk.GetBufferMemoryRequirements, "vkGetBufferMemoryRequirements")
		bind(&vk.AllocateMemory, "vkAllocateMemory")
		bind(&vk.FreeMemory, "vkFreeMemory")
		bind(&vk.BindBufferMemory, "vkBindBufferMemory")
		bind(&vk.MapMemory, "vkMapMemory")
		bind(&vk.UnmapMemory, "vkUnmapMemory")
		bind(&vk.CreateShaderModule, "vkCreateShaderModule")
		bind(&vk.DestroyShaderModule, "vkDestroyShaderModule")
		bind(&vk.CreateDescriptorSetLayout, "vkCreateDescriptorSetLayout")
		bind(&vk.DestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout")
		bind(&vk.CreatePipelineLayout, "vkCreatePipelineLayout")
		bind(&vk.DestroyPipelineLayout, "vkDestroyPipelineLayout")
		bind(&vk.CreateComputePipelines, "vkCreateComputePipelines")
		bind(&vk.DestroyPipeline, "vkDestroyPipeline")
		bind(&vk.CreateDescriptorPool, "vkCreateDescriptorPool")
		bind(&vk.DestroyDescriptorPool, "vkDestroyDescriptorPool")
		bind(&vk.AllocateDescriptorSets, "vkAllocateDescriptorSets")
		bind(&vk.UpdateDescriptorSets, "vkUpdateDescriptorSets")
		bind(&vk.CreateCommandPool, "vkCreateCommandPool")
		bind(&vk.DestroyCommandPool, "vkDestroyCommandPool")
		bind(&vk.AllocateCommandBuffers, "vkAllocateCommandBuffers")
		bind(&vk.FreeCommandBuffers, "vkFreeCommandBuffers")
		bind(&vk.BeginCommandBuffer, "vkBeginCommandBuffer")
		bind(&vk.EndCommandBuffer, "vkEndCommandBuffer")
		bind(&vk.CmdBindPipeline, "vkCmdBindPipeline")
		bind(&vk.CmdBindDescriptorSets, "vkCmdBindDescriptorSets")
		bind(&vk.CmdPushConstants, "vkCmdPushConstants")
		bind(&vk.CmdDispatch, "vkCmdDispatch")
		bind(&vk.CmdPipelineBarrier, "vkCmdPipelineBarrier")
		bind(&vk.CmdCopyBuffer, "vkCmdCopyBuffer")
		bind(&vk.QueueSubmit, "vkQueueSubmit")
		bind(&vk.QueueWaitIdle, "vkQueueWaitIdle")
		bind(&vk.CreateFence, "vkCreateFence")
		bind(&vk.DestroyFence, "vkDestroyFence")
		bind(&vk.WaitForFences, "vkWaitForFences")
		bind(&vk.ResetFences, "vkResetFences")
	})
	return loadErr
}

func check(what string, r int32) error {
	if r != vkSuccess {
		return fmt.Errorf("gpu: %s: VkResult %d", what, r)
	}
	return nil
}

func cstr(s string) *byte {
	b := append([]byte(s), 0)
	return &b[0]
}

func gostr(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
