#pragma once
#include <array>
#include <atomic>
#include <cstddef>

namespace pipeline {

// Lock-free SPSC (single-producer, single-consumer) ring buffer
// Constraint - exactly one producer thread and one consumer thread
// One slot is kept empty (usable capacity = Capacity - 1) to distinguish empty vs full
template <typename T, std::size_t Capacity>
class SpscRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
    static constexpr std::size_t MASK = Capacity - 1;

    // Cache-line separated (alignas(64)) to prevent false sharing between cores
    alignas(64) std::atomic<std::size_t> head_{0}; // Producer-owned write index
    alignas(64) std::atomic<std::size_t> tail_{0}; // Consumer-owned write index
    alignas(64) std::array<T, Capacity> buffer_{};

public:
    // Producer-only
    bool try_push(const T& item) {
        std::size_t head = head_.load(std::memory_order_relaxed);
        std::size_t next = (head + 1) & MASK;
        if (next == tail_.load(std::memory_order_acquire))
            return false;
        buffer_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Consumer-only
    bool try_pop(T& out) {
        std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire))
            return false;
        out = buffer_[tail];
        tail_.store((tail + 1) & MASK, std::memory_order_release);
        return true;
    }

    // Purposefully racy for monitoring, not for control flow
    std::size_t approx_size() const {
        std::size_t h = head_.load(std::memory_order_relaxed);
        std::size_t t = tail_.load(std::memory_order_relaxed);
        return (h - t) & MASK;
    }

    static constexpr std::size_t capacity() { return Capacity - 1; }
};

} // namespace pipeline