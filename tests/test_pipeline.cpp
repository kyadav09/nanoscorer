#include "orderbook/order_book.hpp"
#include "pipeline/order_event.hpp"
#include "pipeline/spsc_ring_buffer.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

using namespace pipeline;
using namespace ob;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "CHECK FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            std::abort(); \
        } \
    } while(0) \

void test_basic_single_threaded() {
    SpscRingBuffer<int, 8> rb;
    CHECK(rb.capacity() == 7); // 1 slot reserved to distinguish full/empty

    int out;
    CHECK(!rb.try_pop(out));

    for (int i = 0; i < 7; ++i) CHECK(rb.try_push(i));
    CHECK(!rb.try_push(999)); // Full

    for (int i = 0; i < 7; ++i) {
        CHECK(rb.try_pop(out));
        CHECK(out == i);
    }
    CHECK(!rb.try_pop(out));

    // Exercise index wraparound across varying batch sizes
    int next_push = 0, next_pop = 0;
    for (int cycle = 0; cycle < 2000; ++cycle) {
        int batch = 1 + (cycle % 6);
        for (int i = 0; i < batch; ++i) {
            CHECK(rb.try_push(next_push));
            ++next_push;
        }
        for (int i = 0; i < batch; ++i) {
            CHECK(rb.try_pop(out));
            CHECK(out == next_pop);
            ++next_pop;
        }
    }
    std::puts("test_basic_single_threaded: OK");
}

// Two-thread stress test: verifies strict FIFO order, zero loss, and zero duplication under contention.
void test_two_thread_stress(int N) {
    SpscRingBuffer<int, 4096> rb;
    std::atomic<bool> producer_done{false};
    std::vector<int> received;
    received.reserve(static_cast<size_t>(N));

    std::thread producer([&] {
        for (int i = 0; i < N; ++i) {
            while (!rb.try_push(i)) std::this_thread::yield();
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&] {
        int out;
        for (;;) {
            if (rb.try_pop(out)) {
                received.push_back(out);
            } else if (producer_done.load(std::memory_order_acquire)) {
                while (rb.try_pop(out)) received.push_back(out);
                break;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    CHECK(static_cast<int>(received.size()) == N);
    for (int i = 0; i < N; ++i) CHECK(received[static_cast<size_t>(i)] == i);
    std::printf("test_two_thread_stress(N=%d): OK -- strict FIFO order, zero loss/dup\n", N);
}

struct LiveOrder {
    OrderId id;
    Qty qty;
};

// Generates a deterministic event stream respecting valid order lifecycle rules (Add/Cancel/Modify).
std::vector<OrderEvent> generate_events(uint32_t seed, int count, int num_levels, Price base) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> op_dist(0, 2);
    std::uniform_int_distribution<int> price_off(0, num_levels - 1);
    std::uniform_int_distribution<Qty> qty_dist(1, 1000);

    std::vector<OrderEvent> events;
    events.reserve(static_cast<size_t>(count));
    std::vector<LiveOrder> live;
    OrderId next_id = 1;

    for (int i = 0; i < count; ++i) {
        int kind = live.empty() ? 0 : op_dist(rng);
        OrderEvent e;
        if (kind == 0) {
            e.type = EventType::Add;
            e.id = next_id++;
            e.side = (rng() & 1) ? Side::Bid : Side::Ask;
            e.price = base + price_off(rng);
            e.qty = qty_dist(rng);
            live.push_back({e.id, e.qty});
        } else if (kind == 1) {
            size_t idx = rng() % live.size();
            e.type = EventType::Cancel;
            e.id = live[idx].id;
            live[idx] = live.back();
            live.pop_back();
        } else {
            size_t idx = rng() % live.size();
            e.type = EventType::Modify;
            e.id = live[idx].id;
            e.qty = 1 + (rng() % live[idx].qty);
            live[idx].qty = e.qty;
        }
        events.push_back(e);
    }
    return events;
}

template <size_t NUM_LEVELS, size_t MAX_ORDERS>
void apply_event(OrderBook<NUM_LEVELS, MAX_ORDERS>& book, const OrderEvent& e) {
    switch (e.type) {
        case EventType::Add: book.add_order(e.id, e.side, e.price, e.qty); break;
        case EventType::Cancel: book.cancel_order(e.id); break;
        case EventType::Modify: book.modify_qty(e.id, e.qty); break;
    }
}

// End-to-end pipeline test: compares single-threaded reference run against two-thread lock-free transport.
void test_order_event_pipeline(uint32_t seed, int num_events) {
    constexpr size_t NUM_LEVELS = 8192;
    constexpr Price BASE = 10000;
    auto events = generate_events(seed, num_events, static_cast<int>(NUM_LEVELS), BASE);

    auto reference = std::make_unique<OrderBook<NUM_LEVELS, 200000>>(BASE);
    for (const auto& e : events) apply_event(*reference, e);
    Price ref_bid_p = 0, ref_ask_p = 0;
    Qty ref_bid_q = 0, ref_ask_q = 0;
    bool ref_has_bid = reference->best_bid(ref_bid_p, ref_bid_q);
    bool ref_has_ask = reference->best_ask(ref_ask_p, ref_ask_q);

    auto consumer_book = std::make_unique<OrderBook<NUM_LEVELS, 200000>>(BASE);
    SpscRingBuffer<OrderEvent, 4096> rb;
    std::atomic<bool> producer_done{false};
    std::vector<OrderEvent> received;
    received.reserve(events.size());

    auto t0 = std::chrono::steady_clock::now();
    std::thread producer([&] {
        for (const auto& e : events) {
            while (!rb.try_push(e)) std::this_thread::yield();
        }
        producer_done.store(true, std::memory_order_release);
    });
    std::thread consumer([&] {
        OrderEvent e;
        for (;;) {
            if (rb.try_pop(e)) {
                apply_event(*consumer_book, e);
                received.push_back(e);
            } else if (producer_done.load(std::memory_order_acquire)) {
                while (rb.try_pop(e)) {
                    apply_event(*consumer_book, e);
                    received.push_back(e);
                }
                break;
            } else {
                std::this_thread::yield();
            }
        }
    });
    producer.join();
    consumer.join();
    auto t1 = std::chrono::steady_clock::now();

    CHECK(received.size() == events.size());
    for (size_t i = 0; i < events.size(); ++i) CHECK(received[i] == events[i]);

    Price cb_bid_p = 0, cb_ask_p = 0;
    Qty cb_bid_q = 0, cb_ask_q = 0;
    bool cb_has_bid = consumer_book->best_bid(cb_bid_p, cb_bid_q);
    bool cb_has_ask = consumer_book->best_ask(cb_ask_p, cb_ask_q);
    CHECK(cb_has_bid == ref_has_bid);
    CHECK(cb_has_ask == ref_has_ask);
    if (ref_has_bid) CHECK(cb_bid_p == ref_bid_p && cb_bid_q == ref_bid_q);
    if (ref_has_ask) CHECK(cb_ask_p == ref_ask_p && cb_ask_q == ref_ask_q);

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("test_order_event_pipeline(seed=%u, events=%d): OK -- sequence exact, book matches reference\n", seed, num_events);
    std::printf("  throughput: %.0f events/sec through 2 threads\n", static_cast<double>(num_events) / secs);
}

// Unpadded variant for architectural comparison against false-sharing overhead.
template <typename T, size_t Capacity>
class UnpaddedSpscRingBuffer {
    static constexpr size_t MASK = Capacity - 1;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    std::array<T, Capacity> buffer_{};

public:
    bool try_push(const T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) & MASK;
        if (next == tail_.load(std::memory_order_acquire)) return false;
        buffer_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }
    bool try_pop(T& out) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return false;
        out = buffer_[tail];
        tail_.store((tail + 1) & MASK, std::memory_order_release);
        return true;
    }
};

template <typename RingBuffer>
double time_stress_run(int N) {
    RingBuffer rb;
    std::atomic<bool> producer_done{false};
    std::atomic<int> received_count{0};

    auto t0 = std::chrono::steady_clock::now();
    std::thread producer([&] {
        for (int i = 0; i < N; ++i) {
            while (!rb.try_push(i)) std::this_thread::yield();
        }
        producer_done.store(true, std::memory_order_release);
    });
    std::thread consumer([&] {
        int out;
        for (;;) {
            if (rb.try_pop(out)) {
                received_count.fetch_add(1, std::memory_order_relaxed);
            } else if (producer_done.load(std::memory_order_acquire)) {
                while (rb.try_pop(out)) received_count.fetch_add(1, std::memory_order_relaxed);
                break;
            } else {
                std::this_thread::yield();
            }
        }
    });
    producer.join();
    consumer.join();
    auto t1 = std::chrono::steady_clock::now();
    (void)received_count;
    return std::chrono::duration<double>(t1 - t0).count();
}

void bench_padding_comparison() {
    constexpr int N = 2000000;
    double padded_s = time_stress_run<SpscRingBuffer<int, 4096>>(N);
    double unpadded_s = time_stress_run<UnpaddedSpscRingBuffer<int, 4096>>(N);
    std::printf("padded:   %.0f events/sec\n", N / padded_s);
    std::printf("unpadded: %.0f events/sec\n", N / unpadded_s);
    std::puts("(single core sandbox -- false sharing requires multiple cores to manifest)");
}

int main() {
    test_basic_single_threaded();
    test_two_thread_stress(500000);
    test_order_event_pipeline(42, 200000);
    bench_padding_comparison();
    std::puts("all tests passed");
    return 0;
}