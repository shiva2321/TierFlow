#include "tf_io.h"

#ifdef _WIN32
#include <windows.h>
#include <ioringapi.h>

#include <algorithm>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace tierflow {

namespace {

struct IoRingApi {
    using fn_QueryIoRingCapabilities = HRESULT(WINAPI*)(IORING_CAPABILITIES*);
    using fn_CreateIoRing            = HRESULT(WINAPI*)(IORING_VERSION, IORING_CREATE_FLAGS, UINT32, UINT32, HIORING*);
    using fn_CloseIoRing             = HRESULT(WINAPI*)(HIORING);
    using fn_SubmitIoRing            = HRESULT(WINAPI*)(HIORING, UINT32, UINT32, UINT32*);
    using fn_PopIoRingCompletion     = HRESULT(WINAPI*)(HIORING, IORING_CQE*);
    using fn_BuildIoRingReadFile     = HRESULT(WINAPI*)(HIORING, IORING_HANDLE_REF, IORING_BUFFER_REF, UINT32, UINT64, UINT_PTR, IORING_SQE_FLAGS);
    using fn_BuildIoRingCancelReq    = HRESULT(WINAPI*)(HIORING, IORING_HANDLE_REF, UINT_PTR, UINT_PTR);

    HMODULE                    dll = nullptr;
    fn_QueryIoRingCapabilities query_caps = nullptr;
    fn_CreateIoRing            create     = nullptr;
    fn_CloseIoRing             close      = nullptr;
    fn_SubmitIoRing            submit     = nullptr;
    fn_PopIoRingCompletion     pop_cqe    = nullptr;
    fn_BuildIoRingReadFile     build_read = nullptr;
    fn_BuildIoRingCancelReq    build_cancel = nullptr;
    bool                       loaded     = false;
    IORING_CAPABILITIES        caps{};

    static const IoRingApi& instance() {
        static IoRingApi api;
        return api;
    }

private:
    IoRingApi() {
        dll = LoadLibraryW(L"kernelbase.dll");
        if (!dll) return;
        query_caps   = reinterpret_cast<fn_QueryIoRingCapabilities>(GetProcAddress(dll, "QueryIoRingCapabilities"));
        create       = reinterpret_cast<fn_CreateIoRing>(GetProcAddress(dll, "CreateIoRing"));
        close        = reinterpret_cast<fn_CloseIoRing>(GetProcAddress(dll, "CloseIoRing"));
        submit       = reinterpret_cast<fn_SubmitIoRing>(GetProcAddress(dll, "SubmitIoRing"));
        pop_cqe      = reinterpret_cast<fn_PopIoRingCompletion>(GetProcAddress(dll, "PopIoRingCompletion"));
        build_read   = reinterpret_cast<fn_BuildIoRingReadFile>(GetProcAddress(dll, "BuildIoRingReadFile"));
        build_cancel = reinterpret_cast<fn_BuildIoRingCancelReq>(GetProcAddress(dll, "BuildIoRingCancelRequest"));

        if (query_caps && create && close && submit && pop_cqe && build_read && build_cancel) {
            if (SUCCEEDED(query_caps(&caps)) && caps.MaxVersion >= IORING_VERSION_1) {
                loaded = true;
            }
        }
    }
};

class tf_io_ioring final : public tf_io_ex {
public:
    explicit tf_io_ioring(const tf_io_options& opts) : opts_(opts) {
        if (opts_.max_inflight == 0) opts_.max_inflight = 32;
        if (opts_.max_inflight > 256) opts_.max_inflight = 256;
        if (opts_.align_bytes == 0) opts_.align_bytes = 4096;
        if (opts_.strict_priority) {
            if (opts_.prefetch_inflight_cap == 0) opts_.prefetch_inflight_cap = 1;
            if (opts_.prefetch_inflight_cap > opts_.max_inflight) {
                opts_.prefetch_inflight_cap = opts_.max_inflight;
            }
        } else {
            opts_.prefetch_inflight_cap = opts_.max_inflight;
        }
        slots_.resize(opts_.max_inflight);
    }

    ~tf_io_ioring() override { close(); }

    bool open(const char* path) override {
        std::lock_guard<std::mutex> lock(mu_);
        close_locked();
        const auto& api = IoRingApi::instance();
        if (!api.loaded || !path || path[0] == '\0') return false;

        int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
        if (wlen <= 0) return false;
        std::wstring wpath(static_cast<size_t>(wlen), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath.data(), wlen);

        HANDLE hf = CreateFileW(
            wpath.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
            nullptr);
        if (hf == INVALID_HANDLE_VALUE) {
            stats_.last_error = GetLastError();
            return false;
        }

        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(hf, &sz)) {
            stats_.last_error = GetLastError();
            CloseHandle(hf);
            return false;
        }

        const UINT32 sq_sz = std::max<UINT32>(64, static_cast<UINT32>(opts_.max_inflight * 4));
        const UINT32 cq_sz = std::max<UINT32>(128, static_cast<UINT32>(opts_.max_inflight * 4));
        IORING_CREATE_FLAGS flags{IORING_CREATE_REQUIRED_FLAGS_NONE, IORING_CREATE_ADVISORY_FLAGS_NONE};
        IORING_VERSION ver = std::min<IORING_VERSION>(api.caps.MaxVersion, IORING_VERSION_3);

        HIORING ring = nullptr;
        HRESULT hr   = api.create(ver, flags, sq_sz, cq_sz, &ring);
        if (FAILED(hr) || !ring) {
            stats_.last_error = static_cast<uint32_t>(hr);
            CloseHandle(hf);
            return false;
        }

        hfile_     = hf;
        ring_      = ring;
        file_size_ = static_cast<uint64_t>(sz.QuadPart);
        stats_     = tf_io_stats{};
        return true;
    }

    void close() override {
        std::lock_guard<std::mutex> lock(mu_);
        close_locked();
    }

    bool submit(const tf_read_req& req) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (hfile_ == INVALID_HANDLE_VALUE || !ring_) {
            ++stats_.rejected_requests;
            return false;
        }

        const size_t align = opts_.align_bytes;
        if (req.dst == nullptr || req.size == 0 || req.size > 0x7FFFF000ULL) {
            ++stats_.rejected_requests;
            return false;
        }
        if ((req.offset % align) != 0 || (req.size % align) != 0 ||
            (reinterpret_cast<uintptr_t>(req.dst) % align) != 0) {
            ++stats_.rejected_requests;
            return false;
        }
        if (req.offset > file_size_ || req.size > (file_size_ - req.offset)) {
            ++stats_.rejected_requests;
            return false;
        }
        if (tag_active_locked(req.tag)) {
            ++stats_.rejected_requests;
            return false;
        }

        drain_cqes_locked();

        if (opts_.strict_priority) {
            if (req.priority >= TF_PRIO_DEMAND) {
                demand_q_.push_back(req);
                ++stats_.submitted_demand;
            } else {
                prefetch_q_.push_back(req);
                ++stats_.submitted_prefetch;
            }
        } else {
            fifo_q_.push_back(req);
            if (req.priority >= TF_PRIO_DEMAND) ++stats_.submitted_demand;
            else ++stats_.submitted_prefetch;
        }

        dispatch_locked();
        return true;
    }

    int poll(tf_read_req* done, int max) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (hfile_ == INVALID_HANDLE_VALUE || !ring_ || max <= 0 || !done) return 0;

        drain_cqes_locked();
        dispatch_locked();
        drain_cqes_locked();
        dispatch_locked();

        int count = 0;
        while (count < max && !ready_done_.empty()) {
            done[count++] = ready_done_.front();
            ready_done_.pop_front();
        }
        return count;
    }

    bool cancel(uint64_t tag) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (hfile_ == INVALID_HANDLE_VALUE || !ring_) return false;
        const auto& api = IoRingApi::instance();

        auto erase_from_queue = [tag](std::deque<tf_read_req>& q) -> bool {
            for (auto it = q.begin(); it != q.end(); ++it) {
                if (it->tag == tag) {
                    q.erase(it);
                    return true;
                }
            }
            return false;
        };

        if (erase_from_queue(prefetch_q_) || erase_from_queue(demand_q_) || erase_from_queue(fifo_q_)) {
            ++stats_.cancelled_requests;
            return true;
        }

        for (size_t i = 0; i < slots_.size(); ++i) {
            RingSlot& slot = slots_[i];
            if (!slot.in_use || slot.req.tag != tag || slot.cancel_requested) continue;

            slot.cancel_requested = true;
            const UINT_PTR read_ud   = slot.user_data;
            const UINT_PTR cancel_ud = kCancelMarker | read_ud;
            api.build_cancel(ring_, IoRingHandleRefFromHandle(hfile_), read_ud, cancel_ud);
            UINT32 submitted = 0;
            api.submit(ring_, 0, 0, &submitted);

            while (slot.in_use) {
                IORING_CQE cqe{};
                if (api.pop_cqe(ring_, &cqe) == S_OK) {
                    handle_cqe_locked(cqe);
                } else {
                    UINT32 sub2 = 0;
                    api.submit(ring_, 1, INFINITE, &sub2);
                }
            }
            dispatch_locked();
            return true;
        }

        return false;
    }

    size_t alignment() const override { return opts_.align_bytes; }

    uint64_t file_size() const override {
        std::lock_guard<std::mutex> lock(mu_);
        return file_size_;
    }

    tf_io_stats stats() const override {
        std::lock_guard<std::mutex> lock(mu_);
        tf_io_stats s = stats_;
        s.queued_demand   = demand_q_.size();
        s.queued_prefetch = prefetch_q_.size();
        return s;
    }

    const char* backend_name() const override { return "win32_ioring"; }

    void drain() override {
        std::lock_guard<std::mutex> lock(mu_);
        const auto& api = IoRingApi::instance();
        if (hfile_ == INVALID_HANDLE_VALUE || !ring_) return;
        while (!demand_q_.empty() || !prefetch_q_.empty() || !fifo_q_.empty() ||
               stats_.inflight_demand > 0 || stats_.inflight_prefetch > 0) {
            dispatch_locked();
            drain_cqes_locked();
            if (stats_.inflight_demand > 0 || stats_.inflight_prefetch > 0) {
                UINT32 sub = 0;
                api.submit(ring_, 1, 10, &sub);
                drain_cqes_locked();
            }
        }
        ready_done_.clear();
    }

private:
    static constexpr UINT_PTR kCancelMarker = (UINT_PTR(1) << 62);

    struct RingSlot {
        tf_read_req req{};
        UINT_PTR    user_data        = 0;
        bool        in_use           = false;
        bool        cancel_requested = false;
    };

    void close_locked() {
        const auto& api = IoRingApi::instance();
        if (ring_ && (stats_.inflight_demand > 0 || stats_.inflight_prefetch > 0)) {
            if (hfile_ != INVALID_HANDLE_VALUE) {
                CancelIoEx(hfile_, nullptr);
            }
            for (int tries = 0; tries < 200 && (stats_.inflight_demand > 0 || stats_.inflight_prefetch > 0); ++tries) {
                UINT32 sub = 0;
                api.submit(ring_, 1, 5, &sub);
                drain_cqes_locked();
            }
        }
        if (ring_) {
            api.close(ring_);
            ring_ = nullptr;
        }
        if (hfile_ != INVALID_HANDLE_VALUE) {
            CloseHandle(hfile_);
            hfile_ = INVALID_HANDLE_VALUE;
        }
        demand_q_.clear();
        prefetch_q_.clear();
        fifo_q_.clear();
        ready_done_.clear();
        for (auto& s : slots_) {
            s.in_use           = false;
            s.cancel_requested = false;
        }
        file_size_ = 0;
    }

    bool tag_active_locked(uint64_t tag) const {
        for (const auto& r : demand_q_)   if (r.tag == tag) return true;
        for (const auto& r : prefetch_q_) if (r.tag == tag) return true;
        for (const auto& r : fifo_q_)     if (r.tag == tag) return true;
        for (const auto& s : slots_)      if (s.in_use && s.req.tag == tag) return true;
        return false;
    }

    size_t active_inflight_locked() const {
        return stats_.inflight_demand + stats_.inflight_prefetch;
    }

    int find_free_slot_locked() const {
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (!slots_[i].in_use) return static_cast<int>(i);
        }
        return -1;
    }

    bool queue_sqe_locked(const tf_read_req& req, int idx) {
        const auto& api = IoRingApi::instance();
        RingSlot& slot  = slots_[static_cast<size_t>(idx)];
        slot.req              = req;
        slot.user_data        = static_cast<UINT_PTR>(idx + 1);
        slot.in_use           = true;
        slot.cancel_requested = false;

        HRESULT hr = api.build_read(
            ring_,
            IoRingHandleRefFromHandle(hfile_),
            IoRingBufferRefFromPointer(req.dst),
            static_cast<UINT32>(req.size),
            req.offset,
            slot.user_data,
            IOSQE_FLAGS_NONE);
        if (FAILED(hr)) {
            slot.in_use       = false;
            stats_.last_error = static_cast<uint32_t>(hr);
            ++stats_.io_errors;
            return false;
        }
        if (req.priority >= TF_PRIO_DEMAND) ++stats_.inflight_demand;
        else ++stats_.inflight_prefetch;
        return true;
    }

    void dispatch_locked() {
        const auto& api = IoRingApi::instance();
        UINT32 queued_sqes = 0;

        if (opts_.strict_priority) {
            while (!demand_q_.empty() && active_inflight_locked() < opts_.max_inflight) {
                int idx = find_free_slot_locked();
                if (idx < 0) break;
                tf_read_req req = demand_q_.front();
                demand_q_.pop_front();
                if (queue_sqe_locked(req, idx)) ++queued_sqes;
            }
            while (demand_q_.empty() && !prefetch_q_.empty() &&
                   active_inflight_locked() < opts_.max_inflight &&
                   stats_.inflight_prefetch < opts_.prefetch_inflight_cap) {
                int idx = find_free_slot_locked();
                if (idx < 0) break;
                tf_read_req req = prefetch_q_.front();
                prefetch_q_.pop_front();
                if (queue_sqe_locked(req, idx)) ++queued_sqes;
            }
        } else {
            while (!fifo_q_.empty() && active_inflight_locked() < opts_.max_inflight) {
                int idx = find_free_slot_locked();
                if (idx < 0) break;
                tf_read_req req = fifo_q_.front();
                fifo_q_.pop_front();
                if (queue_sqe_locked(req, idx)) ++queued_sqes;
            }
        }

        if (queued_sqes > 0) {
            UINT32 submitted = 0;
            HRESULT hr = api.submit(ring_, 0, 0, &submitted);
            if (FAILED(hr)) {
                stats_.last_error = static_cast<uint32_t>(hr);
                ++stats_.io_errors;
            }
        }
    }

    void handle_cqe_locked(const IORING_CQE& cqe) {
        if ((cqe.UserData & kCancelMarker) != 0) {
            // Cancel-request completion itself; ignore.
            return;
        }
        if (cqe.UserData == 0 || cqe.UserData > slots_.size()) return;
        RingSlot& slot = slots_[static_cast<size_t>(cqe.UserData - 1)];
        if (!slot.in_use) return;

        const bool is_demand = (slot.req.priority >= TF_PRIO_DEMAND);
        if (is_demand && stats_.inflight_demand > 0) --stats_.inflight_demand;
        if (!is_demand && stats_.inflight_prefetch > 0) --stats_.inflight_prefetch;

        const bool was_cancelled = slot.cancel_requested;
        tf_read_req completed    = slot.req;
        slot.in_use             = false;
        slot.cancel_requested   = false;

        if (was_cancelled || cqe.ResultCode == HRESULT_FROM_WIN32(ERROR_OPERATION_ABORTED)) {
            ++stats_.cancelled_requests;
            return;
        }

        if (SUCCEEDED(cqe.ResultCode) && static_cast<uint64_t>(cqe.Information) == completed.size) {
            stats_.bytes_completed += static_cast<uint64_t>(cqe.Information);
            if (is_demand) ++stats_.completed_demand;
            else ++stats_.completed_prefetch;
            ready_done_.push_back(completed);
        } else {
            stats_.last_error = static_cast<uint32_t>(cqe.ResultCode);
            ++stats_.io_errors;
        }
    }

    void drain_cqes_locked() {
        const auto& api = IoRingApi::instance();
        IORING_CQE cqe{};
        while (api.pop_cqe(ring_, &cqe) == S_OK) {
            handle_cqe_locked(cqe);
        }
    }

    tf_io_options           opts_{};
    mutable std::mutex      mu_;
    HANDLE                  hfile_     = INVALID_HANDLE_VALUE;
    HIORING                 ring_      = nullptr;
    uint64_t                file_size_ = 0;
    std::vector<RingSlot>   slots_;
    std::deque<tf_read_req> demand_q_;
    std::deque<tf_read_req> prefetch_q_;
    std::deque<tf_read_req> fifo_q_;
    std::deque<tf_read_req> ready_done_;
    tf_io_stats             stats_{};
};

} // namespace

bool tf_ioring_is_supported(std::string* reason_out) {
    const auto& api = IoRingApi::instance();
    if (!api.loaded) {
        if (reason_out) *reason_out = "IoRing API exports unavailable or MaxVersion < IORING_VERSION_1";
        return false;
    }
    if (reason_out) {
        *reason_out = "IoRing supported (MaxVersion=" + std::to_string(static_cast<int>(api.caps.MaxVersion)) +
                      ", MaxSQ=" + std::to_string(api.caps.MaxSubmissionQueueSize) +
                      ", MaxCQ=" + std::to_string(api.caps.MaxCompletionQueueSize) + ")";
    }
    return true;
}

std::unique_ptr<tf_io_ex> tf_io_create_ioring(const tf_io_options& opts) {
    if (!tf_ioring_is_supported(nullptr)) return nullptr;
    return std::make_unique<tf_io_ioring>(opts);
}

} // namespace tierflow
#endif
