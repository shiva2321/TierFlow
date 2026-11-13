#pragma once

#include <cstdint>
#include <string>

namespace tierflow {

struct tf_mem_snapshot {
    uint64_t physical_total_bytes      = 0;
    uint64_t physical_available_bytes  = 0;
    uint64_t system_used_bytes         = 0; // PhysicalTotal - PhysicalAvailable
    uint64_t system_cache_bytes        = 0; // PerformanceInfo SystemCache * PageSize
    uint64_t standby_normal_bytes      = 0; // PDH \Memory\Standby Cache Normal Priority Bytes
    uint64_t modified_page_bytes       = 0; // PDH \Memory\Modified Page List Bytes
    uint64_t pdh_cache_bytes           = 0; // PDH \Memory\Cache Bytes
    uint64_t process_private_bytes     = 0; // GetProcessMemoryInfo PrivateUsage
    uint64_t process_working_set_bytes = 0; // GetProcessMemoryInfo WorkingSetSize
    uint64_t gpu_used_bytes            = 0; // NVML device 0 memory used (0 if unavailable)
    uint64_t gpu_total_bytes           = 0;
    bool     pdh_valid                 = false;
    bool     nvml_valid                = false;
};

class tf_mem_sampler {
public:
    tf_mem_sampler();
    ~tf_mem_sampler();

    tf_mem_sampler(const tf_mem_sampler&) = delete;
    tf_mem_sampler& operator=(const tf_mem_sampler&) = delete;

    tf_mem_snapshot sample();

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// Optional PDH physical disk read bytes counter helper
class tf_disk_counter {
public:
    tf_disk_counter();
    ~tf_disk_counter();

    tf_disk_counter(const tf_disk_counter&) = delete;
    tf_disk_counter& operator=(const tf_disk_counter&) = delete;

    bool   available() const { return valid_; }
    double sample_read_bytes_per_sec(); // returns < 0 if unavailable or priming

private:
    struct Impl;
    Impl* impl_ = nullptr;
    bool  valid_ = false;
};

} // namespace tierflow

