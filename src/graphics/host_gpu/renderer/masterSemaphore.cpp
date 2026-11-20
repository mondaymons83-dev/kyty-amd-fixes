#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/guest_gpu/graphicsRun.h"

#include <atomic>
#include <chrono>

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	m_graphics.device.destroySemaphore(m_semaphore, nullptr);
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

// KYTY_UPLOAD_STATS: wall time spent in semaphore waits, split by the guest GPU thread and the rest.
std::atomic<uint64_t> g_stats_wait_ns_gpu_thread {0};
std::atomic<uint64_t> g_stats_wait_ns_other {0};
std::atomic<uint64_t> g_stats_waits_gpu_thread {0};
std::atomic<uint64_t> g_stats_waits_other {0};

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	const auto wait_start = std::chrono::steady_clock::now();
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	const auto waited = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                              std::chrono::steady_clock::now() - wait_start)
	                                              .count());
	if (GuestGpu::IsGpuThread()) {
		g_stats_wait_ns_gpu_thread.fetch_add(waited, std::memory_order_relaxed);
		g_stats_waits_gpu_thread.fetch_add(1, std::memory_order_relaxed);
	} else {
		g_stats_wait_ns_other.fetch_add(waited, std::memory_order_relaxed);
		g_stats_waits_other.fetch_add(1, std::memory_order_relaxed);
	}
	Refresh();
}

} // namespace Libs::Graphics
