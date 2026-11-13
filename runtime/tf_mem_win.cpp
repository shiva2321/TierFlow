#include "tf_mem_win.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#include <pdhmsg.h>

namespace tierflow {

namespace {
struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};
using fn_nvmlInit_v2             = int (*)(void);
using fn_nvmlShutdown            = int (*)(void);
using fn_nvmlDeviceGetHandleByIdx = int (*)(unsigned int, void**);
using fn_nvmlDeviceGetMemoryInfo = int (*)(void*, nvmlMemory_t*);
} // namespace

struct tf_mem_sampler::Impl {
    PDH_HQUERY   pdh_query        = nullptr;
    PDH_HCOUNTER standby_counter  = nullptr;
    PDH_HCOUNTER modified_counter = nullptr;
    PDH_HCOUNTER cache_counter    = nullptr;
    bool         pdh_ok           = false;

    HMODULE                     nvml_dll = nullptr;
    void*                       nvml_dev = nullptr;
    fn_nvmlDeviceGetMemoryInfo  nvml_mem = nullptr;
    fn_nvmlShutdown             nvml_sd  = nullptr;
    bool                        nvml_ok  = false;
};

tf_mem_sampler::tf_mem_sampler() : impl_(new Impl()) {
    if (PdhOpenQueryW(nullptr, 0, &impl_->pdh_query) == ERROR_SUCCESS) {
        PDH_STATUS s1 = PdhAddEnglishCounterW(impl_->pdh_query, L"\\Memory\\Standby Cache Normal Priority Bytes", 0, &impl_->standby_counter);
        PDH_STATUS s2 = PdhAddEnglishCounterW(impl_->pdh_query, L"\\Memory\\Modified Page List Bytes", 0, &impl_->modified_counter);
        PDH_STATUS s3 = PdhAddEnglishCounterW(impl_->pdh_query, L"\\Memory\\Cache Bytes", 0, &impl_->cache_counter);
        if (s1 == ERROR_SUCCESS && s2 == ERROR_SUCCESS && s3 == ERROR_SUCCESS) {
            PdhCollectQueryData(impl_->pdh_query);
            impl_->pdh_ok = true;
        }
    }

    impl_->nvml_dll = LoadLibraryW(L"nvml.dll");
    if (!impl_->nvml_dll) {
        impl_->nvml_dll = LoadLibraryW(L"C:\\Windows\\System32\\nvml.dll");
    }
    if (impl_->nvml_dll) {
        auto init_fn = reinterpret_cast<fn_nvmlInit_v2>(GetProcAddress(impl_->nvml_dll, "nvmlInit_v2"));
        auto handle_fn = reinterpret_cast<fn_nvmlDeviceGetHandleByIdx>(GetProcAddress(impl_->nvml_dll, "nvmlDeviceGetHandleByIndex_v2"));
        impl_->nvml_mem = reinterpret_cast<fn_nvmlDeviceGetMemoryInfo>(GetProcAddress(impl_->nvml_dll, "nvmlDeviceGetMemoryInfo"));
        impl_->nvml_sd  = reinterpret_cast<fn_nvmlShutdown>(GetProcAddress(impl_->nvml_dll, "nvmlShutdown"));
        if (init_fn && handle_fn && impl_->nvml_mem && init_fn() == 0) {
            if (handle_fn(0, &impl_->nvml_dev) == 0) {
                impl_->nvml_ok = true;
            }
        }
    }
}

tf_mem_sampler::~tf_mem_sampler() {
    if (impl_) {
        if (impl_->pdh_query) PdhCloseQuery(impl_->pdh_query);
        if (impl_->nvml_ok && impl_->nvml_sd) impl_->nvml_sd();
        if (impl_->nvml_dll) FreeLibrary(impl_->nvml_dll);
        delete impl_;
        impl_ = nullptr;
    }
}

tf_mem_snapshot tf_mem_sampler::sample() {
    tf_mem_snapshot snap{};

    PERFORMANCE_INFORMATION pi{};
    pi.cb = sizeof(pi);
    if (GetPerformanceInfo(&pi, sizeof(pi))) {
        const uint64_t page_sz       = static_cast<uint64_t>(pi.PageSize);
        snap.physical_total_bytes     = static_cast<uint64_t>(pi.PhysicalTotal) * page_sz;
        snap.physical_available_bytes = static_cast<uint64_t>(pi.PhysicalAvailable) * page_sz;
        snap.system_used_bytes        = snap.physical_total_bytes - snap.physical_available_bytes;
        snap.system_cache_bytes       = static_cast<uint64_t>(pi.SystemCache) * page_sz;
    }

    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        snap.process_private_bytes     = static_cast<uint64_t>(pmc.PrivateUsage);
        snap.process_working_set_bytes = static_cast<uint64_t>(pmc.WorkingSetSize);
    }

    if (impl_ && impl_->pdh_ok && PdhCollectQueryData(impl_->pdh_query) == ERROR_SUCCESS) {
        PDH_FMT_COUNTERVALUE v{};
        if (PdhGetFormattedCounterValue(impl_->standby_counter, PDH_FMT_LARGE, nullptr, &v) == ERROR_SUCCESS) {
            snap.standby_normal_bytes = static_cast<uint64_t>(v.largeValue);
        }
        if (PdhGetFormattedCounterValue(impl_->modified_counter, PDH_FMT_LARGE, nullptr, &v) == ERROR_SUCCESS) {
            snap.modified_page_bytes = static_cast<uint64_t>(v.largeValue);
        }
        if (PdhGetFormattedCounterValue(impl_->cache_counter, PDH_FMT_LARGE, nullptr, &v) == ERROR_SUCCESS) {
            snap.pdh_cache_bytes = static_cast<uint64_t>(v.largeValue);
        }
        snap.pdh_valid = true;
    }

    if (impl_ && impl_->nvml_ok && impl_->nvml_mem && impl_->nvml_dev) {
        nvmlMemory_t m{};
        if (impl_->nvml_mem(impl_->nvml_dev, &m) == 0) {
            snap.gpu_used_bytes  = static_cast<uint64_t>(m.used);
            snap.gpu_total_bytes = static_cast<uint64_t>(m.total);
            snap.nvml_valid      = true;
        }
    }

    return snap;
}

struct tf_disk_counter::Impl {
    PDH_HQUERY   query   = nullptr;
    PDH_HCOUNTER counter = nullptr;
    bool         primed  = false;
};

tf_disk_counter::tf_disk_counter() : impl_(new Impl()) {
    if (PdhOpenQueryW(nullptr, 0, &impl_->query) == ERROR_SUCCESS) {
        if (PdhAddEnglishCounterW(impl_->query, L"\\PhysicalDisk(_Total)\\Disk Read Bytes/sec", 0, &impl_->counter) == ERROR_SUCCESS) {
            if (PdhCollectQueryData(impl_->query) == ERROR_SUCCESS) {
                impl_->primed = true;
                valid_        = true;
            }
        }
    }
}

tf_disk_counter::~tf_disk_counter() {
    if (impl_) {
        if (impl_->query) PdhCloseQuery(impl_->query);
        delete impl_;
        impl_ = nullptr;
    }
}

double tf_disk_counter::sample_read_bytes_per_sec() {
    if (!valid_ || !impl_) return -1.0;
    if (PdhCollectQueryData(impl_->query) != ERROR_SUCCESS) return -1.0;
    PDH_FMT_COUNTERVALUE v{};
    if (PdhGetFormattedCounterValue(impl_->counter, PDH_FMT_DOUBLE, nullptr, &v) != ERROR_SUCCESS) return -1.0;
    return v.doubleValue;
}

} // namespace tierflow
#endif

