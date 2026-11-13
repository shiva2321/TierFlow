#include "tf_io.h"

#if defined(__linux__)
#include <errno.h>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#ifndef O_DIRECT
#define O_DIRECT 00040000
#endif

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

constexpr uint64_t kCancelMarker = 0xFFFFFFFFFFFFFFFEULL;

inline int sys_io_uring_setup(unsigned entries, struct io_uring_params* params) {
#ifdef __NR_io_uring_setup
    return static_cast<int>(syscall(__NR_io_uring_setup, entries, params));
#else
    (void)entries;
    (void)params;
    errno = ENOSYS;
    return -1;
#endif
}

inline int sys_io_uring_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags) {
#ifdef __NR_io_uring_enter
    return static_cast<int>(syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, nullptr, 0));
#else
    (void)fd;
    (void)to_submit;
    (void)min_complete;
    (void)flags;
    errno = ENOSYS;
    return -1;
#endif
}

inline int sys_io_uring_register(int fd, unsigned opcode, const void* arg, unsigned nr_args) {
#ifdef __NR_io_uring_register
    return static_cast<int>(syscall(__NR_io_uring_register, fd, opcode, arg, nr_args));
#else
    (void)fd;
    (void)opcode;
    (void)arg;
    (void)nr_args;
    errno = ENOSYS;
    return -1;
#endif
}

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
        if (fd >= 0) {
            o_direct_ = true;
        } else if (errno == EINVAL || errno == ENOTSUP || errno == EOPNOTSUPP) {
            // Filesystems such as tmpfs, overlayfs, or 9P/drvfs reject O_DIRECT with EINVAL
            fd = ::open(path, O_RDONLY);
            o_direct_ = false;
        }
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

        init_uring_locked();

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
        if (!done || max <= 0) return 0;
        std::lock_guard<std::mutex> lock(mu_);
        return poll_locked(done, max);
    }

    int wait(tf_read_req* done, int max, uint32_t timeout_ms) override {
        if (!done || max <= 0) return 0;
        std::unique_lock<std::mutex> lock(mu_);
        int count = poll_locked(done, max);
        if (count > 0 || !use_ring_ || ring_fd_ < 0 ||
            (stats_.inflight_demand == 0 && stats_.inflight_prefetch == 0 &&
             demand_q_.empty() && prefetch_q_.empty())) {
            return count;
        }

        const int rfd = ring_fd_;
        lock.unlock();
        struct pollfd pfd{};
        pfd.fd     = rfd;
        pfd.events = POLLIN;
        ::poll(&pfd, 1, static_cast<int>(timeout_ms));
        lock.lock();
        return poll_locked(done, max);
    }

    bool cancel(uint64_t tag) override {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto it = demand_q_.begin(); it != demand_q_.end(); ++it) {
            if (it->tag == tag) {
                demand_q_.erase(it);
                ++stats_.cancelled_requests;
                return true;
            }
        }
        for (auto it = prefetch_q_.begin(); it != prefetch_q_.end(); ++it) {
            if (it->tag == tag) {
                prefetch_q_.erase(it);
                ++stats_.cancelled_requests;
                return true;
            }
        }
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].in_use && slots_[i].req.tag == tag) {
                slots_[i].cancel_requested = true;
                if (use_ring_ && ring_fd_ >= 0) {
                    struct io_uring_sqe* sqe = acquire_sqe_locked();
                    if (sqe) {
                        std::memset(sqe, 0, sizeof(*sqe));
                        sqe->opcode    = IORING_OP_ASYNC_CANCEL;
                        sqe->addr      = static_cast<uint64_t>(i + 1);
                        sqe->user_data = kCancelMarker;
                        commit_sqe_locked(1);
                    }
                    // Wait until the kernel CQE retires this slot so req.dst is never
                    // mutated after cancel() returns true to the caller.
                    drain_cqes_locked();
                    while (slots_[i].in_use) {
                        sys_io_uring_enter(ring_fd_, 0, 1, IORING_ENTER_GETEVENTS);
                        drain_cqes_locked();
                    }
                }
                return true;
            }
        }
        return false;
    }

    void drain() override {
        std::lock_guard<std::mutex> lock(mu_);
        drain_cqes_locked();
        dispatch_locked();
        if (use_ring_ && ring_fd_ >= 0) {
            while (stats_.inflight_demand > 0 || stats_.inflight_prefetch > 0 ||
                   !demand_q_.empty() || !prefetch_q_.empty()) {
                sys_io_uring_enter(ring_fd_, 0, 1, IORING_ENTER_GETEVENTS);
                drain_cqes_locked();
                dispatch_locked();
            }
        }
        ready_done_.clear();
    }

    size_t alignment() const override { return opts_.align_bytes; }
    uint64_t file_size() const override { return file_size_; }

    tf_io_stats stats() const override {
        std::lock_guard<std::mutex> lock(mu_);
        tf_io_stats out = stats_;
        out.queued_demand   = demand_q_.size();
        out.queued_prefetch = prefetch_q_.size();
        return out;
    }

    const char* backend_name() const override {
        if (use_ring_ && reg_buffers_ok_ && o_direct_) return "linux_io_uring_direct_reg";
        if (use_ring_ && o_direct_) return "linux_io_uring_direct";
        if (use_ring_) return "linux_io_uring_buffered";
        return o_direct_ ? "linux_pread_direct" : "linux_pread_buffered";
    }

private:
    struct Slot {
        tf_read_req req{};
        bool        in_use           = false;
        bool        cancel_requested = false;
    };

    void init_uring_locked() {
        const unsigned req_entries = static_cast<unsigned>(std::max<size_t>(64, opts_.max_inflight * 4));
        struct io_uring_params p{};
        int rfd = sys_io_uring_setup(req_entries, &p);
        if (rfd < 0) {
            // Gracefully fall back to pread under seccomp / unsupported kernel
            use_ring_ = false;
            return;
        }

        sq_ring_sz_ = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
        cq_ring_sz_ = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
        if (p.features & IORING_FEAT_SINGLE_MMAP) {
            if (cq_ring_sz_ > sq_ring_sz_) sq_ring_sz_ = cq_ring_sz_;
            cq_ring_sz_ = sq_ring_sz_;
        }

        void* sq_ptr = mmap(nullptr, sq_ring_sz_, PROT_READ | PROT_WRITE,
                            MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_SQ_RING);
        if (sq_ptr == MAP_FAILED) {
            ::close(rfd);
            use_ring_ = false;
            return;
        }

        void* cq_ptr = sq_ptr;
        bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
        if (!single) {
            cq_ptr = mmap(nullptr, cq_ring_sz_, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_CQ_RING);
            if (cq_ptr == MAP_FAILED) {
                munmap(sq_ptr, sq_ring_sz_);
                ::close(rfd);
                use_ring_ = false;
                return;
            }
        }

        sqes_sz_ = p.sq_entries * sizeof(struct io_uring_sqe);
        void* sqes_map = mmap(nullptr, sqes_sz_, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_SQES);
        if (sqes_map == MAP_FAILED) {
            if (!single) munmap(cq_ptr, cq_ring_sz_);
            munmap(sq_ptr, sq_ring_sz_);
            ::close(rfd);
            use_ring_ = false;
            return;
        }

        ring_fd_     = rfd;
        sq_ring_ptr_ = sq_ptr;
        cq_ring_ptr_ = cq_ptr;
        single_mmap_ = single;
        sqes_        = static_cast<struct io_uring_sqe*>(sqes_map);

        auto* sb = static_cast<uint8_t*>(sq_ring_ptr_);
        auto* cb = static_cast<uint8_t*>(cq_ring_ptr_);
        sq_head_  = reinterpret_cast<uint32_t*>(sb + p.sq_off.head);
        sq_tail_  = reinterpret_cast<uint32_t*>(sb + p.sq_off.tail);
        sq_mask_  = *reinterpret_cast<uint32_t*>(sb + p.sq_off.ring_mask);
        sq_array_ = reinterpret_cast<uint32_t*>(sb + p.sq_off.array);

        cq_head_  = reinterpret_cast<uint32_t*>(cb + p.cq_off.head);
        cq_tail_  = reinterpret_cast<uint32_t*>(cb + p.cq_off.tail);
        cq_mask_  = *reinterpret_cast<uint32_t*>(cb + p.cq_off.ring_mask);
        cqes_     = reinterpret_cast<struct io_uring_cqe*>(cb + p.cq_off.cqes);
        use_ring_ = true;

        // Register fixed scratch page to verify IORING_REGISTER_BUFFERS capability
        reg_buf_ = tf_aligned_alloc(4096, 4096);
        if (reg_buf_) {
            struct iovec iov{};
            iov.iov_base = reg_buf_;
            iov.iov_len  = 4096;
            if (sys_io_uring_register(ring_fd_, IORING_REGISTER_BUFFERS, &iov, 1) == 0) {
                reg_buffers_ok_ = true;
            } else {
                tf_aligned_free(reg_buf_);
                reg_buf_ = nullptr;
            }
        }
    }

    void close_locked() {
        if (use_ring_ && ring_fd_ >= 0) {
            if (reg_buffers_ok_) {
                sys_io_uring_register(ring_fd_, IORING_UNREGISTER_BUFFERS, nullptr, 0);
                reg_buffers_ok_ = false;
            }
            if (reg_buf_) {
                tf_aligned_free(reg_buf_);
                reg_buf_ = nullptr;
            }
            if (sqes_) {
                munmap(sqes_, sqes_sz_);
                sqes_ = nullptr;
            }
            if (cq_ring_ptr_ && !single_mmap_) {
                munmap(cq_ring_ptr_, cq_ring_sz_);
            }
            if (sq_ring_ptr_) {
                munmap(sq_ring_ptr_, sq_ring_sz_);
            }
            sq_ring_ptr_ = nullptr;
            cq_ring_ptr_ = nullptr;
            ::close(ring_fd_);
            ring_fd_  = -1;
            use_ring_ = false;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        for (auto& s : slots_) {
            s = Slot{};
        }
        file_size_ = 0;
        o_direct_  = false;
        demand_q_.clear();
        prefetch_q_.clear();
        ready_done_.clear();
    }

    struct io_uring_sqe* acquire_sqe_locked() {
        const uint32_t head = __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE);
        const uint32_t tail = *sq_tail_;
        if (tail - head >= (sq_mask_ + 1)) {
            return nullptr;
        }
        const uint32_t idx = tail & sq_mask_;
        sq_array_[idx] = idx;
        __atomic_store_n(sq_tail_, tail + 1, __ATOMIC_RELEASE);
        return &sqes_[idx];
    }

    void commit_sqe_locked(unsigned n_sub) {
        if (n_sub > 0 && use_ring_ && ring_fd_ >= 0) {
            sys_io_uring_enter(ring_fd_, n_sub, 0, 0);
        }
    }

    int find_free_slot_locked() const {
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (!slots_[i].in_use) return static_cast<int>(i);
        }
        return -1;
    }

    void dispatch_locked() {
        if (use_ring_ && ring_fd_ >= 0) {
            unsigned submitted = 0;
            while (!demand_q_.empty()) {
                int idx = find_free_slot_locked();
                if (idx < 0) break;
                struct io_uring_sqe* sqe = acquire_sqe_locked();
                if (!sqe) break;
                slots_[static_cast<size_t>(idx)].req              = demand_q_.front();
                slots_[static_cast<size_t>(idx)].in_use           = true;
                slots_[static_cast<size_t>(idx)].cancel_requested = false;
                demand_q_.pop_front();
                std::memset(sqe, 0, sizeof(*sqe));
                sqe->opcode    = IORING_OP_READ;
                sqe->fd        = fd_;
                sqe->off       = slots_[static_cast<size_t>(idx)].req.offset;
                sqe->addr      = reinterpret_cast<uintptr_t>(slots_[static_cast<size_t>(idx)].req.dst);
                sqe->len       = static_cast<uint32_t>(slots_[static_cast<size_t>(idx)].req.size);
                sqe->user_data = static_cast<uint64_t>(idx + 1);
                ++stats_.inflight_demand;
                ++submitted;
            }
            while (demand_q_.empty() && !prefetch_q_.empty() &&
                   stats_.inflight_prefetch < opts_.prefetch_inflight_cap) {
                int idx = find_free_slot_locked();
                if (idx < 0) break;
                struct io_uring_sqe* sqe = acquire_sqe_locked();
                if (!sqe) break;
                slots_[static_cast<size_t>(idx)].req              = prefetch_q_.front();
                slots_[static_cast<size_t>(idx)].in_use           = true;
                slots_[static_cast<size_t>(idx)].cancel_requested = false;
                prefetch_q_.pop_front();
                std::memset(sqe, 0, sizeof(*sqe));
                sqe->opcode    = IORING_OP_READ;
                sqe->fd        = fd_;
                sqe->off       = slots_[static_cast<size_t>(idx)].req.offset;
                sqe->addr      = reinterpret_cast<uintptr_t>(slots_[static_cast<size_t>(idx)].req.dst);
                sqe->len       = static_cast<uint32_t>(slots_[static_cast<size_t>(idx)].req.size);
                sqe->user_data = static_cast<uint64_t>(idx + 1);
                ++stats_.inflight_prefetch;
                ++submitted;
            }
            if (submitted > 0) {
                commit_sqe_locked(submitted);
            }
            return;
        }

        // Synchronous pread fallback when io_uring is blocked (e.g., seccomp / legacy kernel)
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
    }

    void handle_cqe_locked(const struct io_uring_cqe* cqe) {
        const uint64_t ud = cqe->user_data;
        if (ud == kCancelMarker || ud == 0 || ud > slots_.size()) return;
        Slot& s = slots_[static_cast<size_t>(ud - 1)];
        if (!s.in_use) return;
        const bool is_demand = (s.req.priority >= TF_PRIO_DEMAND);
        if (is_demand && stats_.inflight_demand > 0) --stats_.inflight_demand;
        if (!is_demand && stats_.inflight_prefetch > 0) --stats_.inflight_prefetch;
        const bool was_cancelled = s.cancel_requested;
        tf_read_req completed    = s.req;
        s.in_use                 = false;
        s.cancel_requested       = false;
        if (was_cancelled || cqe->res == -ECANCELED) {
            ++stats_.cancelled_requests;
            return;
        }
        if (cqe->res == static_cast<int32_t>(completed.size)) {
            stats_.bytes_completed += static_cast<uint64_t>(cqe->res);
            if (is_demand) ++stats_.completed_demand;
            else ++stats_.completed_prefetch;
            ready_done_.push_back(completed);
        } else {
            ++stats_.io_errors;
        }
    }

    void drain_cqes_locked() {
        if (!use_ring_ || !cq_head_ || !cq_tail_) return;
        uint32_t head = *cq_head_;
        while (head != __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE)) {
            const struct io_uring_cqe* cqe = &cqes_[head & cq_mask_];
            handle_cqe_locked(cqe);
            ++head;
        }
        __atomic_store_n(cq_head_, head, __ATOMIC_RELEASE);
    }

    int poll_locked(tf_read_req* done, int max) {
        drain_cqes_locked();
        dispatch_locked();
        drain_cqes_locked();

        int out = 0;
        while (out < max && !ready_done_.empty()) {
            done[out++] = ready_done_.front();
            ready_done_.pop_front();
        }
        return out;
    }

    tf_io_options           opts_{};
    mutable std::mutex      mu_;
    int                     fd_             = -1;
    uint64_t                file_size_      = 0;
    bool                    o_direct_       = false;

    bool                    use_ring_       = false;
    bool                    single_mmap_    = false;
    bool                    reg_buffers_ok_ = false;
    void*                   reg_buf_        = nullptr;
    int                     ring_fd_        = -1;
    void*                   sq_ring_ptr_    = nullptr;
    void*                   cq_ring_ptr_    = nullptr;
    size_t                  sq_ring_sz_     = 0;
    size_t                  cq_ring_sz_     = 0;
    struct io_uring_sqe*    sqes_           = nullptr;
    size_t                  sqes_sz_        = 0;
    uint32_t*               sq_head_        = nullptr;
    uint32_t*               sq_tail_        = nullptr;
    uint32_t                sq_mask_        = 0;
    uint32_t*               sq_array_       = nullptr;
    uint32_t*               cq_head_        = nullptr;
    uint32_t*               cq_tail_        = nullptr;
    uint32_t                cq_mask_        = 0;
    struct io_uring_cqe*    cqes_           = nullptr;

    std::vector<Slot>       slots_;
    std::deque<tf_read_req> demand_q_;
    std::deque<tf_read_req> prefetch_q_;
    std::deque<tf_read_req> ready_done_;
    tf_io_stats             stats_{};
};

} // namespace

bool tf_ioring_is_supported(std::string* reason_out) {
#ifdef __NR_io_uring_setup
    struct io_uring_params p{};
    int rfd = sys_io_uring_setup(8, &p);
    if (rfd >= 0) {
        ::close(rfd);
        if (reason_out) *reason_out = "Linux io_uring supported via kernel syscalls";
        return true;
    }
    if (reason_out) {
        *reason_out = std::string("io_uring_setup failed errno=") + std::to_string(errno);
    }
    return false;
#else
    if (reason_out) *reason_out = "__NR_io_uring_setup unavailable";
    return false;
#endif
}

std::unique_ptr<tf_io_ex> tf_io_create_ioring(const tf_io_options& opts) {
    return std::make_unique<tf_io_linux>(opts);
}

std::unique_ptr<tf_io_ex> tf_io_create_iocp(const tf_io_options&) {
    return nullptr;
}

std::unique_ptr<tf_io_ex> tf_io_create_auto(const tf_io_options& opts) {
    return std::make_unique<tf_io_linux>(opts);
}

} // namespace tierflow
#endif
