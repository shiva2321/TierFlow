#include "tf_io.h"

#if defined(__linux__)
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(TF_HAS_LIBURING)
#include <liburing.h>
#endif

#include <algorithm>
#include <deque>
#include <mutex>
#include <vector>

namespace tierflow {

void* tf_aligned_alloc(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, std::max<size_t>(alignment, sizeof(void*)), bytes) != 0) {
        return nullptr;
    }
    return ptr;
}

void tf_aligned_free(void* ptr) {
    free(ptr);
}

namespace {

class tf_io_linux final : public tf_io_ex {
public:
    explicit tf_io_linux(const tf_io_options& opts) : opts_(opts) {
        if (opts_.max_inflight == 0) opts_.max_inflight = 32;
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

    ~tf_io_linux() override { close(); }

    bool open(const char* path) override {
        std::lock_guard<std::mutex> lock(mu_);
        close_locked();
        if (!path) return false;
        int fd = ::open(path, O_RDONLY | O_DIRECT);
        if (fd < 0) {
            stats_.last_error = static_cast<uint32_t>(errno);
            return false;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            stats_.last_error = static_cast<uint32_t>(errno);
            ::close(fd);
            return false;
        }
#if defined(TF_HAS_LIBURING)
        if (io_uring_queue_init(static_cast<unsigned>(opts_.max_inflight * 2), &ring_, 0) != 0) {
            ::close(fd);
            return false;
        }
        ring_init_ = true;
#endif
        fd_        = fd;
        file_size_ = static_cast<uint64_t>(st.st_size);
        stats_     = tf_io_stats{};
        return true;
    }

    void close() override {
        std::lock_guard<std::mutex> lock(mu_);
        close_locked();
    }

    bool submit(const tf_read_req& req) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (fd_ < 0 || !req.dst || req.size == 0) {
            ++stats_.rejected_requests;
            return false;
        }
        const size_t align = opts_.align_bytes;
        if ((req.offset % align) != 0 || (req.size % align) != 0 ||
            (reinterpret_cast<uintptr_t>(req.dst) % align) != 0) {
            ++stats_.rejected_requests;
            return false;
        }
        if (req.offset > file_size_ || req.size > (file_size_ - req.offset)) {
            ++stats_.rejected_requests;
            return false;
        }
        if (req.priority >= TF_PRIO_DEMAND) {
            demand_q_.push_back(req);
            ++stats_.submitted_demand;
        } else {
            prefetch_q_.push_back(req);
            ++stats_.submitted_prefetch;
        }
        dispatch_locked();
        return true;
    }

    int poll(tf_read_req* done, int max) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (fd_ < 0 || max <= 0 || !done) return 0;
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
        for (auto it = prefetch_q_.begin(); it != prefetch_q_.end(); ++it) {
            if (it->tag == tag) {
                prefetch_q_.erase(it);
                ++stats_.cancelled_requests;
                return true;
            }
        }
        for (auto it = demand_q_.begin(); it != demand_q_.end(); ++it) {
            if (it->tag == tag) {
                demand_q_.erase(it);
                ++stats_.cancelled_requests;
                return true;
            }
        }
#if defined(TF_HAS_LIBURING)
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].in_use && slots_[i].req.tag == tag && !slots_[i].cancel_requested) {
                slots_[i].cancel_requested = true;
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
                if (sqe) {
                    io_uring_prep_cancel64(sqe, i + 1, 0);
                    io_uring_sqe_set_data64(sqe, kCancelMarker);
                    io_uring_submit(&ring_);
                }
                while (slots_[i].in_use) {
                    struct io_uring_cqe* cqe = nullptr;
                    if (io_uring_wait_cqe(&ring_, &cqe) == 0 && cqe) {
                        handle_cqe_locked(cqe);
                        io_uring_cqe_seen(&ring_, cqe);
                    } else {
                        break;
                    }
                }
                return true;
            }
        }
#endif
        return false;
    }

    size_t alignment() const override { return opts_.align_bytes; }

    uint64_t file_size() const override {
        std::lock_guard<std::mutex> lock(mu_);
        return file_size_;
    }

    tf_io_stats stats() const override {
        std::lock_guard<std::mutex> lock(mu_);
        return stats_;
    }

    const char* backend_name() const override {
#if defined(TF_HAS_LIBURING)
        return "linux_io_uring_odirect";
#else
        return "linux_pread_odirect";
#endif
    }

    void drain() override {
        std::lock_guard<std::mutex> lock(mu_);
        while (!demand_q_.empty() || !prefetch_q_.empty() ||
               stats_.inflight_demand > 0 || stats_.inflight_prefetch > 0) {
            dispatch_locked();
            drain_cqes_locked();
        }
        ready_done_.clear();
    }

private:
    static constexpr uint64_t kCancelMarker = (uint64_t(1) << 62);
    struct Slot {
        tf_read_req req{};
        bool        in_use           = false;
        bool        cancel_requested = false;
    };

    void close_locked() {
#if defined(TF_HAS_LIBURING)
        if (ring_init_) {
            io_uring_queue_exit(&ring_);
            ring_init_ = false;
        }
#endif
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        file_size_ = 0;
        demand_q_.clear();
        prefetch_q_.clear();
        ready_done_.clear();
    }

    void dispatch_locked() {
#if defined(TF_HAS_LIBURING)
        unsigned submitted = 0;
        while (!demand_q_.empty()) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            if (!sqe) break;
            int idx = -1;
            for (size_t i = 0; i < slots_.size(); ++i) if (!slots_[i].in_use) { idx = static_cast<int>(i); break; }
            if (idx < 0) break;
            slots_[idx].req = demand_q_.front();
            slots_[idx].in_use = true;
            slots_[idx].cancel_requested = false;
            demand_q_.pop_front();
            io_uring_prep_read(sqe, fd_, slots_[idx].req.dst, static_cast<unsigned>(slots_[idx].req.size), slots_[idx].req.offset);
            io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(idx + 1));
            ++stats_.inflight_demand;
            ++submitted;
        }
        while (demand_q_.empty() && !prefetch_q_.empty() && stats_.inflight_prefetch < opts_.prefetch_inflight_cap) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            if (!sqe) break;
            int idx = -1;
            for (size_t i = 0; i < slots_.size(); ++i) if (!slots_[i].in_use) { idx = static_cast<int>(i); break; }
            if (idx < 0) break;
            slots_[idx].req = prefetch_q_.front();
            slots_[idx].in_use = true;
            slots_[idx].cancel_requested = false;
            prefetch_q_.pop_front();
            io_uring_prep_read(sqe, fd_, slots_[idx].req.dst, static_cast<unsigned>(slots_[idx].req.size), slots_[idx].req.offset);
            io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(idx + 1));
            ++stats_.inflight_prefetch;
            ++submitted;
        }
        if (submitted > 0) io_uring_submit(&ring_);
#else
        auto run_req = [&](const tf_read_req& req) {
            ssize_t n = ::pread(fd_, req.dst, req.size, static_cast<off_t>(req.offset));
            if (n == static_cast<ssize_t>(req.size)) {
                stats_.bytes_completed += static_cast<uint64_t>(n);
                if (req.priority >= TF_PRIO_DEMAND) ++stats_.completed_demand;
                else ++stats_.completed_prefetch;
                ready_done_.push_back(req);
            } else {
                ++stats_.io_errors;
            }
        };
        while (!demand_q_.empty()) {
            tf_read_req r = demand_q_.front();
            demand_q_.pop_front();
            run_req(r);
        }
        while (!prefetch_q_.empty()) {
            tf_read_req r = prefetch_q_.front();
            prefetch_q_.pop_front();
            run_req(r);
        }
#endif
    }

#if defined(TF_HAS_LIBURING)
    void handle_cqe_locked(struct io_uring_cqe* cqe) {
        uint64_t ud = io_uring_cqe_get_data64(cqe);
        if (ud == kCancelMarker || ud == 0 || ud > slots_.size()) return;
        Slot& s = slots_[ud - 1];
        if (!s.in_use) return;
        const bool is_demand = (s.req.priority >= TF_PRIO_DEMAND);
        if (is_demand && stats_.inflight_demand > 0) --stats_.inflight_demand;
        if (!is_demand && stats_.inflight_prefetch > 0) --stats_.inflight_prefetch;
        const bool was_cancelled = s.cancel_requested;
        tf_read_req completed = s.req;
        s.in_use = false;
        s.cancel_requested = false;
        if (was_cancelled || cqe->res == -ECANCELED) {
            ++stats_.cancelled_requests;
            return;
        }
        if (cqe->res == static_cast<int>(completed.size)) {
            stats_.bytes_completed += static_cast<uint64_t>(cqe->res);
            if (is_demand) ++stats_.completed_demand;
            else ++stats_.completed_prefetch;
            ready_done_.push_back(completed);
        } else {
            ++stats_.io_errors;
        }
    }
#endif

    void drain_cqes_locked() {
#if defined(TF_HAS_LIBURING)
        struct io_uring_cqe* cqe = nullptr;
        while (io_uring_peek_cqe(&ring_, &cqe) == 0 && cqe) {
            handle_cqe_locked(cqe);
            io_uring_cqe_seen(&ring_, cqe);
        }
#endif
    }

    tf_io_options           opts_{};
    mutable std::mutex      mu_;
    int                     fd_        = -1;
    uint64_t                file_size_ = 0;
#if defined(TF_HAS_LIBURING)
    struct io_uring         ring_{};
    bool                    ring_init_ = false;
#endif
    std::vector<Slot>       slots_;
    std::deque<tf_read_req> demand_q_;
    std::deque<tf_read_req> prefetch_q_;
    std::deque<tf_read_req> ready_done_;
    tf_io_stats             stats_{};
};

} // namespace

bool tf_ioring_is_supported(std::string* reason_out) {
    if (reason_out) *reason_out = "Windows IoRing not available on Linux";
    return false;
}

std::unique_ptr<tf_io_ex> tf_io_create_ioring(const tf_io_options&) { return nullptr; }
std::unique_ptr<tf_io_ex> tf_io_create_iocp(const tf_io_options&)   { return nullptr; }
std::unique_ptr<tf_io_ex> tf_io_create_auto(const tf_io_options& opts) {
    return std::make_unique<tf_io_linux>(opts);
}

} // namespace tierflow
#endif

