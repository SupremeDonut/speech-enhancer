#pragma once

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace speech {

// Exactly one producer owns head and one consumer owns tail.
template <class T>
class SPSCRingBuffer final {
public:
    explicit SPSCRingBuffer(std::size_t capacity)
        : storage_(checked_capacity(capacity)), capacity_(capacity), mask_(capacity-1) {}
    SPSCRingBuffer(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;

    std::size_t write(const T* source, std::size_t count) noexcept {
        const auto head = head_.load(std::memory_order_relaxed);
        const auto tail = tail_.load(std::memory_order_acquire);
        const auto available = capacity_ - (head-tail);
        const auto amount = count < available ? count : available;
        for (std::size_t i=0; i<amount; ++i) storage_[(head+i)&mask_] = source[i];
        head_.store(head+amount, std::memory_order_release);
        return amount;
    }
    std::size_t read(T* destination, std::size_t count) noexcept {
        const auto tail = tail_.load(std::memory_order_relaxed);
        const auto head = head_.load(std::memory_order_acquire);
        const auto amount = (head-tail) < count ? (head-tail) : count;
        for (std::size_t i=0; i<amount; ++i) destination[i] = storage_[(tail+i)&mask_];
        tail_.store(tail+amount, std::memory_order_release);
        return amount;
    }
    std::size_t readable() const noexcept {
        const auto tail = tail_.load(std::memory_order_relaxed);
        const auto head = head_.load(std::memory_order_acquire);
        return head-tail;
    }
    std::size_t writable() const noexcept {
        const auto head = head_.load(std::memory_order_relaxed);
        const auto tail = tail_.load(std::memory_order_acquire);
        return capacity_-(head-tail);
    }
    void discard_all() noexcept {
        const auto head = head_.load(std::memory_order_acquire);
        tail_.store(head, std::memory_order_release);
    }
    constexpr std::size_t capacity() const noexcept { return capacity_; }

private:
    static std::size_t checked_capacity(std::size_t capacity) {
        if (capacity < 2 || (capacity & (capacity-1)) != 0)
            throw std::invalid_argument("SPSC capacity must be a power of two >= 2");
        return capacity;
    }
    std::vector<T> storage_;
    const std::size_t capacity_;
    const std::size_t mask_;
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

}  // namespace speech
