#ifndef KYTY_COMMON_PERFSTATS_H_
#define KYTY_COMMON_PERFSTATS_H_

#include "common/common.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

// Lightweight timing summary printed to the console every few seconds.
// Enabled by setting the environment variable KYTY_PERF_STATS=1.
namespace PerfStats {

enum class Section : uint32_t {
	ShaderCompile,     // SPIR-V generation and shader module creation
	PipelineCreate,    // vkCreate{Graphics,Compute}Pipelines
	SrtMaterialize,    // per-draw shader resource evaluation
	GpuProcess,        // guest GPU command processing
	QueueSubmit,       // vkQueueSubmit, including the queue lock wait
	Present,           // swapchain present, including the queue lock wait
	ColorMetaReadback, // synchronous DCC/CMask metadata readback before binding a color image
	GpuWait,           // GPU thread blocked until the host GPU reaches a submission
	SyncReadback,      // BufferCache::ReadMemory (submit, wait and copy back)
	CpFlush,           // guest command buffer boundaries that submit host work
	DrawPrepare,       // per-draw render state, targets and resources
	DrawExecute,       // per-draw binding commit and command recording
	Dispatch,          // compute dispatch preparation and recording
	FlushSlice,        // command buffer flush at the end of a processed submission slice
	ReadbackCpuRead,   // guest CPU read of GPU-written memory (page fault)
	ReadbackCpuWrite,  // guest CPU write to GPU-written memory (page fault)
	ReadbackIndirect,  // indirect dispatch arguments read back on the CPU
	DrawIndirectGpu,   // indexed indirect draws whose GPU-written arguments stay on the GPU
	Count
};

bool Enabled();
void Add(Section section, uint64_t nanos);
// Attributes a wait to its call stack; the slowest sites are printed with the summary.
void AddWaitSite(uint64_t nanos);
// Called once per presented guest frame; prints the summary when the interval elapses.
void FramePresented();

// On-screen HUD data: frame pacing over the last half second, recent frame times (oldest first)
// and the busiest sections of the last printed summary.
struct HudSnapshot {
	double                   fps          = 0.0;
	double                   frame_avg_ms = 0.0;
	double                   frame_max_ms = 0.0;
	std::vector<float>       frame_times_ms;
	std::vector<std::string> sections;
};
HudSnapshot GetHud();

class Scope {
public:
	explicit Scope(Section section)
	    : m_section(section), m_active(Enabled()),
	      m_start(m_active ? std::chrono::steady_clock::now()
	                       : std::chrono::steady_clock::time_point {}) {}
	~Scope() {
		if (m_active) {
			Add(m_section, static_cast<uint64_t>(
			                   std::chrono::duration_cast<std::chrono::nanoseconds>(
			                       std::chrono::steady_clock::now() - m_start)
			                       .count()));
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Section                               m_section;
	bool                                  m_active;
	std::chrono::steady_clock::time_point m_start;
};

} // namespace PerfStats

#define KYTY_PERF_SCOPE(section)                                                                   \
	PerfStats::Scope KYTY_PERF_CONCAT(kyty_perf_scope_, __LINE__)(PerfStats::Section::section)
#define KYTY_PERF_CONCAT_IMPL(a, b) a##b
#define KYTY_PERF_CONCAT(a, b)      KYTY_PERF_CONCAT_IMPL(a, b)

#endif /* KYTY_COMMON_PERFSTATS_H_ */
