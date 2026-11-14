#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tierflow {

constexpr uint8_t TF_PRIO_PREFETCH = 0;
constexpr uint8_t TF_PRIO_DEMAND   = 1;

struct tf_read_req {
    uint32_t object_id = 0;
    uint64_t offset    = 0;
    uint64_t size      = 0;
    void*    dst       = nullptr;
    uint8_t  priority  = TF_PRIO_DEMAND;
    uint64_t tag       = 0;
};

class tf_io {
public:
    virtual ~tf_io() = default;
    virtual bool   open(const char* path) = 0;           // unbuffered
    virtual bool   submit(const tf_read_req&) = 0;       // non-blocking
    virtual int    poll(tf_read_req* done, int max) = 0; // completed requests
    virtual bool   cancel(uint64_t tag) = 0;             // best-effort for low-priority reads
    virtual size_t alignment() const = 0;                // sector/page alignment of dst/offset/size
};

struct tf_io_options {
    size_t max_inflight          = 32;   // kernel queue depth limit
    size_t prefetch_inflight_cap = 8;    // max simultaneous in-flight prefetch reads when strict_priority=true
    bool   strict_priority       = true; // demand queue bypasses prefetch queue
    size_t align_bytes           = 4096; // page/sector alignment
};

struct tf_io_stats {
    uint64_t submitted_demand    = 0;
    uint64_t submitted_prefetch  = 0;
    uint64_t completed_demand    = 0;
    uint64_t completed_prefetch  = 0;
    uint64_t cancelled_requests  = 0;
    uint64_t rejected_requests   = 0;
    uint64_t io_errors           = 0;
    uint64_t bytes_completed     = 0;
    uint32_t last_error          = 0;
    size_t   inflight_demand     = 0;
    size_t   inflight_prefetch   = 0;
    size_t   queued_demand       = 0;
    size_t   queued_prefetch     = 0;
};

// Extended instrumentation interface implemented by concrete backends
class tf_io_ex : public tf_io {
public:
    virtual void        close() = 0;
    virtual uint64_t    file_size() const = 0;
    virtual tf_io_stats stats() const = 0;
    virtual const char* backend_name() const = 0;
    virtual void        drain() = 0;
    virtual int         wait(tf_read_req* done, int max, uint32_t timeout_ms) {
        (void)timeout_ms;
        return poll(done, max);
    }
};

// Aligned host buffer allocation (VirtualAlloc on Windows, aligned_alloc/posix_memalign on Linux)
void* tf_aligned_alloc(size_t bytes, size_t alignment = 4096);
void  tf_aligned_free(void* ptr);

// Concrete factories
std::unique_ptr<tf_io_ex> tf_io_create_iocp(const tf_io_options& opts = {});
std::unique_ptr<tf_io_ex> tf_io_create_ioring(const tf_io_options& opts = {});
std::unique_ptr<tf_io_ex> tf_io_create_auto(const tf_io_options& opts = {});
bool                      tf_ioring_is_supported(std::string* reason_out = nullptr);

} // namespace tierflow

