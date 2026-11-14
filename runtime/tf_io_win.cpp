#include "tf_io.h"

#ifdef _WIN32
#include <windows.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace tierflow {

void* tf_aligned_alloc(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;
    if (alignment <= 65536) {
        return VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    }
    return _aligned_malloc(bytes, alignment);
}

void tf_aligned_free(void* ptr) {
    if (!ptr) return;
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(ptr, &mbi, sizeof(mbi)) != 0 && mbi.AllocationBase == ptr && mbi.Type == MEM_PRIVATE) {
        VirtualFree(ptr, 0, MEM_RELEASE);
        return;
    }
    _aligned_free(ptr);
}

namespace {

class tf_io_iocp final : public tf_io_ex {
public:
    explicit tf_io_iocp(const tf_io_options& opts) : opts_(opts) {
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
        for (size_t i = 0; i < slots_.size(); ++i) {
            slots_[i].slot_idx = i;
        }
    }

    ~tf_io_iocp() override { close(); }

    bool open(const char* path) override {
        std::lock_guard<std::mutex> lock(mu_);
        close_locked();
        if (!path || path[0] == '\0') return false;

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

        HANDLE iocp = CreateIoCompletionPort(hf, nullptr, 0, 1);
        if (!iocp) {
            stats_.last_error = GetLastError();
            CloseHandle(hf);
            return false;
        }

        hfile_     = hf;
        iocp_      = iocp;
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
        if (hfile_ == INVALID_HANDLE_VALUE) {
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

        drain_completions_locked(0);

        if (opts_.strict_priority) {
            if (req.priority >= TF_PRIO_DEMAND) {
                demand_q_.push_back(req);
                ++stats_.submitted_demand;
            } else {
                prefetch_q_.push_back(req);
                ++stats_.submitted_prefetch;
            }
        } else {
            // Single FIFO queue null control
            fifo_q_.push_back(req);
            if (req.priority >= TF_PRIO_DEMAND) {
                ++stats_.submitted_demand;
            } else {
                ++stats_.submitted_prefetch;
            }
        }

        dispatch_locked();
        return true;
    }

    int poll(tf_read_req* done, int max) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (hfile_ == INVALID_HANDLE_VALUE || max <= 0 || !done) {
            return 0;
        }

        drain_completions_locked(0);
        dispatch_locked();
        drain_completions_locked(0);
        dispatch_locked();

        int count = 0;
        while (count < max && !ready_done_.empty()) {
            done[count++] = ready_done_.front();
            ready_done_.pop_front();
        }
        return count;
    }

    int wait(tf_read_req* done, int max, uint32_t timeout_ms) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (hfile_ == INVALID_HANDLE_VALUE || max <= 0 || !done) {
            return 0;
        }

        drain_completions_locked(0);
        dispatch_locked();
        if (ready_done_.empty() && active_inflight_locked() > 0 && timeout_ms > 0) {
            drain_completions_locked(static_cast<DWORD>(timeout_ms));
            dispatch_locked();
        }

        int count = 0;
        while (count < max && !ready_done_.empty()) {
            done[count++] = ready_done_.front();
            ready_done_.pop_front();
        }
        return count;
    }

    bool cancel(uint64_t tag) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (hfile_ == INVALID_HANDLE_VALUE) return false;

        // 1. Check unissued software queues first (zero OS interaction, zero dst touches)
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

        // 2. Check active kernel OVERLAPPED slots
        for (size_t i = 0; i < slots_.size(); ++i) {
            IoSlot& slot = slots_[i];
            if (!slot.in_use || slot.req.tag != tag || slot.cancel_requested) {
                continue;
            }

            slot.cancel_requested = true;
            CancelIoEx(hfile_, &slot.ov);

            // Wait synchronously for this OVERLAPPED completion packet from IOCP so that
            // after cancel() returns true, the kernel CANNOT write any more bytes into dst.
            while (slot.in_use) {
                DWORD       bytes_xfer = 0;
                ULONG_PTR   key        = 0;
                OVERLAPPED* ov         = nullptr;
                BOOL ok = GetQueuedCompletionStatus(iocp_, &bytes_xfer, &key, &ov, INFINITE);
                if (!ov) {
                    stats_.last_error = GetLastError();
                    ++stats_.io_errors;
                    break;
                }
                DWORD err = ok ? ERROR_SUCCESS : GetLastError();
                handle_completion_locked(ov, bytes_xfer, err);
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

    const char* backend_name() const override { return "win32_iocp"; }

    void drain() override {
        std::lock_guard<std::mutex> lock(mu_);
        if (hfile_ == INVALID_HANDLE_VALUE) return;
        while (!demand_q_.empty() || !prefetch_q_.empty() || !fifo_q_.empty() ||
               stats_.inflight_demand > 0 || stats_.inflight_prefetch > 0) {
            dispatch_locked();
            if (stats_.inflight_demand > 0 || stats_.inflight_prefetch > 0) {
                drain_completions_locked(1);
            }
        }
        ready_done_.clear();
    }

private:
    struct IoSlot {
        OVERLAPPED  ov{}; // must stay valid in-place (vector resized once in ctor)
        tf_read_req req{};
        size_t      slot_idx         = 0;
        bool        in_use           = false;
        bool        cancel_requested = false;
    };

    void close_locked() {
        if (hfile_ != INVALID_HANDLE_VALUE) {
            CancelIoEx(hfile_, nullptr);
            while (stats_.inflight_demand > 0 || stats_.inflight_prefetch > 0) {
                if (!drain_completions_locked(50)) break;
            }
            CloseHandle(hfile_);
            hfile_ = INVALID_HANDLE_VALUE;
        }
        if (iocp_ != nullptr) {
            CloseHandle(iocp_);
            iocp_ = nullptr;
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

    bool issue_slot_locked(const tf_read_req& req, int idx) {
        IoSlot& slot          = slots_[static_cast<size_t>(idx)];
        slot.ov               = OVERLAPPED{};
        slot.ov.Offset        = static_cast<DWORD>(req.offset & 0xFFFFFFFFULL);
        slot.ov.OffsetHigh    = static_cast<DWORD>(req.offset >> 32);
        slot.req              = req;
        slot.in_use           = true;
        slot.cancel_requested = false;

        const bool is_demand = (req.priority >= TF_PRIO_DEMAND);
        if (is_demand) {
            ++stats_.inflight_demand;
        } else {
            ++stats_.inflight_prefetch;
        }

        DWORD bytes_read = 0;
        BOOL  ok         = ReadFile(hfile_, req.dst, static_cast<DWORD>(req.size), &bytes_read, &slot.ov);
        if (!ok) {
            DWORD err = GetLastError();
            if (err != ERROR_IO_PENDING) {
                slot.in_use = false;
                if (is_demand) --stats_.inflight_demand;
                else --stats_.inflight_prefetch;
                stats_.last_error = err;
                ++stats_.io_errors;
                return false;
            }
        }
        return true;
    }

    void dispatch_locked() {
        if (opts_.strict_priority) {
            while (!demand_q_.empty() && active_inflight_locked() < opts_.max_inflight) {
                int idx = find_free_slot_locked();
                if (idx < 0) break;
                tf_read_req req = demand_q_.front();
                demand_q_.pop_front();
                issue_slot_locked(req, idx);
            }
            while (demand_q_.empty() && !prefetch_q_.empty() &&
                   active_inflight_locked() < opts_.max_inflight &&
                   stats_.inflight_prefetch < opts_.prefetch_inflight_cap) {
                int idx = find_free_slot_locked();
                if (idx < 0) break;
                tf_read_req req = prefetch_q_.front();
                prefetch_q_.pop_front();
                issue_slot_locked(req, idx);
            }
        } else {
            while (!fifo_q_.empty() && active_inflight_locked() < opts_.max_inflight) {
                int idx = find_free_slot_locked();
                if (idx < 0) break;
                tf_read_req req = fifo_q_.front();
                fifo_q_.pop_front();
                issue_slot_locked(req, idx);
            }
        }
    }

    void handle_completion_locked(OVERLAPPED* ov, DWORD bytes_xfer, DWORD err) {
        IoSlot* slot = reinterpret_cast<IoSlot*>(ov);
        if (slot < slots_.data() || slot >= slots_.data() + slots_.size() || !slot->in_use) {
            return;
        }

        const bool is_demand = (slot->req.priority >= TF_PRIO_DEMAND);
        if (is_demand && stats_.inflight_demand > 0) --stats_.inflight_demand;
        if (!is_demand && stats_.inflight_prefetch > 0) --stats_.inflight_prefetch;

        const bool was_cancelled = slot->cancel_requested;
        tf_read_req completed    = slot->req;
        slot->in_use             = false;
        slot->cancel_requested   = false;

        if (was_cancelled || err == ERROR_OPERATION_ABORTED) {
            ++stats_.cancelled_requests;
            return;
        }

        if (err == ERROR_SUCCESS && bytes_xfer == completed.size) {
            stats_.bytes_completed += bytes_xfer;
            if (is_demand) ++stats_.completed_demand;
            else ++stats_.completed_prefetch;
            ready_done_.push_back(completed);
        } else {
            stats_.last_error = err != ERROR_SUCCESS ? err : ERROR_HANDLE_EOF;
            ++stats_.io_errors;
        }
    }

    bool drain_completions_locked(DWORD first_timeout_ms) {
        bool drained_any = false;
        DWORD timeout    = first_timeout_ms;
        for (;;) {
            DWORD       bytes_xfer = 0;
            ULONG_PTR   key        = 0;
            OVERLAPPED* ov         = nullptr;
            BOOL ok = GetQueuedCompletionStatus(iocp_, &bytes_xfer, &key, &ov, timeout);
            timeout = 0;
            if (!ov) {
                break;
            }
            drained_any = true;
            DWORD err   = ok ? ERROR_SUCCESS : GetLastError();
            handle_completion_locked(ov, bytes_xfer, err);
        }
        return drained_any;
    }

    tf_io_options          opts_{};
    mutable std::mutex     mu_;
    HANDLE                 hfile_     = INVALID_HANDLE_VALUE;
    HANDLE                 iocp_      = nullptr;
    uint64_t               file_size_ = 0;
    std::vector<IoSlot>    slots_;
    std::deque<tf_read_req> demand_q_;
    std::deque<tf_read_req> prefetch_q_;
    std::deque<tf_read_req> fifo_q_;
    std::deque<tf_read_req> ready_done_;
    tf_io_stats            stats_{};
};

} // namespace

std::unique_ptr<tf_io_ex> tf_io_create_iocp(const tf_io_options& opts) {
    return std::make_unique<tf_io_iocp>(opts);
}

std::unique_ptr<tf_io_ex> tf_io_create_auto(const tf_io_options& opts) {
    // Win32 IOCP with FILE_FLAG_NO_BUFFERING is the universal Windows default;
    // IoRing is explicitly selectable via tf_io_create_ioring() when desired.
    return tf_io_create_iocp(opts);
}

} // namespace tierflow
#endif

