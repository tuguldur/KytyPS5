#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

// Debug switch (KYTY_DEBUG_FAULT_SYNC=1): before a guest thread reads GPU-written memory,
// wait until the GPU thread has processed every queued guest submission, so the readback sees
// all GPU work the guest has issued so far. Waits are bounded so a submission that waits on
// this thread cannot deadlock it.
static bool DebugFaultSync() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DEBUG_FAULT_SYNC");
		const bool  on    = value != nullptr && value[0] != '\0' && value[0] != '0';
		if (on) {
			std::printf("KYTY_DEBUG_FAULT_SYNC: guest read faults wait for GPU idle\n");
		}
		return on;
	}();
	return enabled;
}

static void DebugFaultSyncWait(GuestGpu& gpu, PageFaultAccess access, uint64_t fault_vaddr) {
	constexpr uint32_t            TimeoutMicros = 200'000;
	static std::atomic<uint64_t> waits {0};
	static std::atomic<uint64_t> timeouts {0};
	const auto                   wait_index = waits.fetch_add(1, std::memory_order_relaxed);
	if (gpu.WaitForIdleFor(TimeoutMicros)) {
		if (wait_index % 5000 == 0) {
			std::printf("FaultSync: %llu waits, %llu timeouts\n",
			            static_cast<unsigned long long>(wait_index + 1),
			            static_cast<unsigned long long>(timeouts.load(std::memory_order_relaxed)));
		}
		return;
	}
	const auto timeout_index = timeouts.fetch_add(1, std::memory_order_relaxed);
	if (timeout_index < 40 || timeout_index % 100 == 0) {
		std::printf("FaultSync timeout #%llu: %s addr=0x%llx (GPU not idle after %ums)\n",
		            static_cast<unsigned long long>(timeout_index),
		            access == PageFaultAccess::Write ? "write" : "read",
		            static_cast<unsigned long long>(fault_vaddr), TimeoutMicros / 1000);
	}
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (access == PageFaultAccess::Read && DebugFaultSync() && !GuestGpu::IsGpuThread() &&
	    !CommandScheduler::InDeferredOperation()) {
		DebugFaultSyncWait(GetGpu(), access, fault_vaddr);
	}
	if (access == PageFaultAccess::Write) {
		KYTY_PERF_SCOPE(ReadbackCpuWrite);
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		KYTY_PERF_SCOPE(ReadbackCpuRead);
		if (PerfStats::Enabled()) {
			static std::atomic<uint32_t> logged {0};
			const auto index = logged.fetch_add(1, std::memory_order_relaxed);
			if (index < 40 || index % 200 == 0) {
				const auto [start, end] = m_buffer_cache.GpuDirtyRun(fault_vaddr);
				const bool gpu_thread   = GuestGpu::IsGpuThread();
				const auto header       = gpu_thread ? GuestGpu::CurrentPm4Header() : 0u;
				// The byte-exact GPU-modified ranges belong to the GPU thread.
				const auto page  = fault_vaddr & ~uint64_t {0xfff};
				bool        exact = false;
				std::string ranges;
				std::string writers;
				const auto describe = [&] {
					exact   = m_buffer_cache.HasGpuDirtyBytes(fault_vaddr, 8);
					ranges  = m_buffer_cache.DescribeGpuModified(page, 0x1000);
					writers = m_buffer_cache.DescribeWriters(page, 0x1000);
				};
				if (gpu_thread) {
					describe();
				} else {
					GetGpu().SendCommandSync(describe);
				}
				std::printf("CpuReadFault #%u: %s pm4_op=0x%02x pm4_r=0x%02x addr=0x%llx "
				            "dirty_run=0x%llx+0x%llx exact=%d page_gpu_bytes:%s writers:%s "
				            "stack:%s\n",
				            index, gpu_thread ? "gpu-thread" : "guest-thread",
				            (header >> 8u) & 0xffu, static_cast<uint32_t>(KYTY_PM4_R(header)),
				            static_cast<unsigned long long>(fault_vaddr),
				            static_cast<unsigned long long>(start),
				            static_cast<unsigned long long>(end - start), exact ? 1 : 0,
				            ranges.c_str(), writers.c_str(), PerfStats::FaultStackText().c_str());
			}
		}
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::TryServePolledRead(uint64_t vaddr, void* data, uint64_t size) noexcept {
	constexpr uint32_t PolledThreshold = 64;
	constexpr uint64_t PolledMaxSize   = 16;
	if (GuestGpu::IsGpuThread() || size == 0 || size > PolledMaxSize || !IsMapped(vaddr, size)) {
		return false;
	}
	PolledRange* range = nullptr;
	{
		std::shared_lock lock(m_polled_mutex);
		if (const auto found = m_polled_ranges.find(vaddr);
		    found != m_polled_ranges.end() && found->second->size == size) {
			range = found->second.get();
		}
	}
	if (range == nullptr) {
		// Only loads that keep faulting are treated as polling.
		std::unique_lock lock(m_polled_mutex);
		if (++m_poll_counts[(vaddr << 4u) | size] < PolledThreshold) {
			return false;
		}
		auto& entry = m_polled_ranges[vaddr];
		if (entry == nullptr) {
			entry       = std::make_unique<PolledRange>();
			entry->size = size;
			std::printf("PolledRead: serving 0x%llx+%llu from the backing with async refresh\n",
			            static_cast<unsigned long long>(vaddr),
			            static_cast<unsigned long long>(size));
		}
		if (entry->size != size) {
			return false;
		}
		range = entry.get();
	}
	if (!LibKernel::Memory::TryReadBacking(vaddr, data, size)) {
		return false;
	}
	if (!range->refresh_pending.exchange(true, std::memory_order_acq_rel)) {
		GetGpu().SendCommand([this, vaddr, size, range] {
			m_buffer_cache.RefreshPolledRange(vaddr, size, [range] {
				range->refresh_pending.store(false, std::memory_order_release);
			});
		});
	}
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations())) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
		m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
	});
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
