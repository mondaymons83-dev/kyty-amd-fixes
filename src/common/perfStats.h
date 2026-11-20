#ifndef KYTY_COMMON_PERF_STATS_H_
#define KYTY_COMMON_PERF_STATS_H_

// always-on counters printed every 60 flips when KYTY_PERF_STATS=1.
// Relaxed atomics: cheap enough to bump on hot paths.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Common::PerfStats {

inline std::atomic<uint64_t> g_lds_stalls_avoided {0}; // indirect LDS dispatches not read back
inline std::atomic<uint64_t> g_sync_downloads {0};     // buffer readbacks that waited for the GPU
inline std::atomic<uint64_t> g_sync_wait_ns {0};       // time spent in those waits
inline std::atomic<uint64_t> g_yield_sleeps {0};       // guest yields turned into short sleeps
// Who asks for GPU-written memory back (each may drain the GPU queue):
inline std::atomic<uint64_t> g_rb_fault_read {0};  // guest CPU read a GPU-tracked page
inline std::atomic<uint64_t> g_rb_fault_write {0}; // guest CPU wrote a GPU-tracked page
inline std::atomic<uint64_t> g_rb_gpu_thread {0};  // GPU thread read GPU-dirty memory (PM4 data)
inline std::atomic<uint64_t> g_rb_meta {0};        // color metadata (fast clear) check

// CPU time of the GPU thread's main steps (ns, summed):
inline std::atomic<uint64_t> g_ns_draw {0};      // whole draw (renderer side)
inline std::atomic<uint64_t> g_ns_dispatch {0};  // whole dispatch (renderer side)
inline std::atomic<uint64_t> g_ns_find_image {0}; // TextureCache::FindImage
inline std::atomic<uint64_t> g_ns_pipeline {0};  // graphics/compute pipeline lookups

inline std::atomic<uint64_t> g_prefetches {0};     // async downloads of hot read-back pages
inline std::atomic<uint64_t> g_prefetch_hits {0};  // guest reads served by one (no full drain)
inline std::atomic<uint64_t> g_stale_reads {0}; // KYTY_ASYNC_READBACK: fault reads served stale

// Where the expensive events land: page (4 KiB) -> count, printed and cleared periodically.
struct PageTally {
	std::mutex                             lock;
	std::unordered_map<uint64_t, uint64_t> counts;
	void Add(uint64_t vaddr) {
		std::scoped_lock guard(lock);
		++counts[vaddr >> 12u];
	}
	// "page:count ..." of the busiest pages, then clears.
	std::string TakeTop(size_t limit) {
		std::vector<std::pair<uint64_t, uint64_t>> items;
		{
			std::scoped_lock guard(lock);
			items.assign(counts.begin(), counts.end());
			counts.clear();
		}
		std::sort(items.begin(), items.end(),
		          [](const auto& a, const auto& b) { return a.second > b.second; });
		std::string out;
		char        text[48];
		for (size_t i = 0; i < items.size() && i < limit; ++i) {
			std::snprintf(text, sizeof(text), " %llx:%llu",
			              static_cast<unsigned long long>(items[i].first << 12u),
			              static_cast<unsigned long long>(items[i].second));
			out += text;
		}
		return out;
	}
};
inline PageTally g_pages_sync_download; // buffer ranges downloaded with a full GPU wait
inline PageTally g_pages_fault_write;   // guest CPU writes into GPU-tracked pages
inline PageTally g_pages_fault_read;    // guest CPU reads of GPU-tracked pages

inline void Bump(std::atomic<uint64_t>& counter, uint64_t value = 1) noexcept {
	counter.fetch_add(value, std::memory_order_relaxed);
}

class ScopedNs final {
public:
	explicit ScopedNs(std::atomic<uint64_t>& counter) noexcept
	    : m_counter(counter), m_start(std::chrono::steady_clock::now()) {}
	~ScopedNs() {
		Bump(m_counter, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                          std::chrono::steady_clock::now() - m_start)
		                                          .count()));
	}
	ScopedNs(const ScopedNs&)            = delete;
	ScopedNs& operator=(const ScopedNs&) = delete;

private:
	std::atomic<uint64_t>&                m_counter;
	std::chrono::steady_clock::time_point m_start;
};

} // namespace Common::PerfStats

#endif /* KYTY_COMMON_PERF_STATS_H_ */
