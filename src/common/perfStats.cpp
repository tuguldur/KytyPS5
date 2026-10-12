#include "common/perfStats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace PerfStats {
namespace {

constexpr auto     ReportInterval = std::chrono::seconds(5);
constexpr uint64_t SlowFrameNanos = 50'000'000;

constexpr std::array<const char*, static_cast<size_t>(Section::Count)> SectionNames = {
    "shader-compile", "pipeline-create", "srt-materialize", "gpu-process", "queue-submit",
    "present", "color-meta-readback", "gpu-wait", "sync-readback", "cp-flush", "draw-prepare",
    "draw-execute", "dispatch", "flush-slice", "readback-cpu-read",
    "readback-cpu-write", "readback-indirect", "draw-indirect-gpu",
};

struct Counter {
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> total {0};
	std::atomic<uint64_t> max {0};
};

std::array<Counter, static_cast<size_t>(Section::Count)> g_counters;

std::mutex                            g_frame_mutex;
std::chrono::steady_clock::time_point g_interval_start;
std::chrono::steady_clock::time_point g_last_frame;
uint64_t                              g_frames       = 0;
uint64_t                              g_slow_frames  = 0;
uint64_t                              g_max_frame_ns = 0;

constexpr int WaitSiteFrames = 10;
struct WaitSite {
	uint64_t                             count = 0;
	uint64_t                             total = 0;
	std::array<uint64_t, WaitSiteFrames> frames {};
};
std::mutex                   g_wait_site_mutex;
std::map<uint64_t, WaitSite> g_wait_sites;

void PrintWaitSites(uint64_t elapsed_ns) {
	std::vector<WaitSite> sites;
	{
		std::lock_guard lock(g_wait_site_mutex);
		for (const auto& [hash, site]: g_wait_sites) sites.push_back(site);
		g_wait_sites.clear();
	}
	std::sort(sites.begin(), sites.end(),
	          [](const auto& a, const auto& b) { return a.total > b.total; });
	for (size_t i = 0; i < sites.size() && i < 4; i++) {
		std::printf("WaitSite: %llux %.0fms (%.0f%%):", static_cast<unsigned long long>(sites[i].count),
		            static_cast<double>(sites[i].total) / 1e6,
		            100.0 * static_cast<double>(sites[i].total) / static_cast<double>(elapsed_ns));
		for (const auto frame: sites[i].frames) {
			if (frame != 0) std::printf(" +0x%llx", static_cast<unsigned long long>(frame));
		}
		std::printf("\n");
	}
}

double Ms(uint64_t nanos) {
	return static_cast<double>(nanos) / 1e6;
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PERF_STATS");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return enabled;
}

void Add(Section section, uint64_t nanos) {
	auto& counter = g_counters[static_cast<size_t>(section)];
	counter.count.fetch_add(1, std::memory_order_relaxed);
	counter.total.fetch_add(nanos, std::memory_order_relaxed);
	auto max = counter.max.load(std::memory_order_relaxed);
	while (nanos > max && !counter.max.compare_exchange_weak(max, nanos, std::memory_order_relaxed)) {
	}
}

void AddWaitSite(uint64_t nanos) {
#if defined(_WIN32)
	void*      stack[WaitSiteFrames + 2] {};
	const auto captured = RtlCaptureStackBackTrace(2, WaitSiteFrames, stack, nullptr);
	const auto base     = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	WaitSite   site;
	uint64_t   hash = 1469598103934665603ull;
	for (USHORT i = 0; i < captured; i++) {
		const auto address = reinterpret_cast<uint64_t>(stack[i]);
		site.frames[i]     = address >= base ? address - base : address;
		hash               = (hash ^ site.frames[i]) * 1099511628211ull;
	}
	std::lock_guard lock(g_wait_site_mutex);
	auto&           entry = g_wait_sites[hash];
	entry.frames          = site.frames;
	entry.count++;
	entry.total += nanos;
#else
	(void)nanos;
#endif
}

void FramePresented() {
	if (!Enabled()) {
		return;
	}
	const auto      now = std::chrono::steady_clock::now();
	std::lock_guard lock(g_frame_mutex);
	if (g_interval_start == std::chrono::steady_clock::time_point {}) {
		g_interval_start = now;
		g_last_frame     = now;
		return;
	}
	const auto frame_ns = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(now - g_last_frame).count());
	g_last_frame = now;
	g_frames++;
	g_slow_frames += frame_ns > SlowFrameNanos ? 1 : 0;
	g_max_frame_ns = frame_ns > g_max_frame_ns ? frame_ns : g_max_frame_ns;

	const auto elapsed = now - g_interval_start;
	if (elapsed < ReportInterval) {
		return;
	}
	const auto elapsed_ns = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
	std::printf("Perf: %.1fs fps=%.1f frame avg=%.1fms max=%.1fms >50ms=%llu |", Ms(elapsed_ns) / 1e3,
	            static_cast<double>(g_frames) * 1e9 / static_cast<double>(elapsed_ns),
	            Ms(elapsed_ns) / static_cast<double>(g_frames), Ms(g_max_frame_ns),
	            static_cast<unsigned long long>(g_slow_frames));
	for (size_t i = 0; i < g_counters.size(); i++) {
		auto&      counter = g_counters[i];
		const auto count   = counter.count.exchange(0, std::memory_order_relaxed);
		const auto total   = counter.total.exchange(0, std::memory_order_relaxed);
		const auto max     = counter.max.exchange(0, std::memory_order_relaxed);
		std::printf(" %s %llux %.0fms (%.0f%%, max %.1fms)", SectionNames[i],
		            static_cast<unsigned long long>(count), Ms(total),
		            100.0 * static_cast<double>(total) / static_cast<double>(elapsed_ns), Ms(max));
	}
	std::printf("\n");
	PrintWaitSites(elapsed_ns);
	std::fflush(stdout);
	g_interval_start = now;
	g_frames         = 0;
	g_slow_frames    = 0;
	g_max_frame_ns   = 0;
}

} // namespace PerfStats
