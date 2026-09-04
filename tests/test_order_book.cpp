#include "orderbook/order_book.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include <atomic>

using namespace ob;

//Always-on check, independent of NDEBUG
#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "CHECK FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            std::abort(); \
        } \
    } while(0) \

void test_basic() {
    auto book = std::make_unique<OrderBook<4096, 1024>>(/*base_price=*/10000);
    Price p;
    Qty q;
    CHECK(!book->best_bid(p,q));
    CHECK(!book->best_ask(p,q));
    
    CHECK(book->add_order(1, Side::Bid, 10050, 100));
    CHECK(book->add_order(2, Side::Bid, 10040, 50));
    CHECK(book->add_order(3, Side::Ask, 10060, 75));

    CHECK(book->best_bid(p, q) && p == 10050 && q == 100);
    CHECK(book->best_ask(p, q) && p == 10060 && q == 75);
    
    CHECK(book->add_order(4, Side::Bid, 10050, 25)); // joins the 10050 level
    CHECK(book->best_bid(p, q) && p == 10050 && q == 125);

    CHECK(book->cancel_order(1)); // remove the first order at 10050
    CHECK(book->best_bid(p, q) && p == 10050 && q == 25); // order 4 remains

    CHECK(book->cancel_order(4));
    CHECK(book->best_bid(p, q) && p == 10040 && q == 50); // falls through

    CHECK(book->modify_qty(2, 10));
    CHECK(book->best_bid(p, q) && p == 10040 && q == 10);

    CHECK(!book->modify_qty(2, 999)); // increases are rejected
    CHECK(!book->cancel_order(999));  // unknown id
    CHECK(!book->add_order(5, Side::Bid, 999, 1)); // below base_price -> out of range

    std::puts("test_basic: OK");
}

struct RefOrder {
    Side side;
    Price price;
    Qty qty;
};

void test_random_differential(uint32_t seed, int num_ops) {
    constexpr std::size_t NUM_LEVELS = 8192;
    constexpr std::size_t MAX_ORDERS = 20000;
    constexpr Price BASE = 10000;

    auto book = std::make_unique<OrderBook<NUM_LEVELS, MAX_ORDERS>>(BASE);

    std::unordered_map<OrderId, RefOrder> ref;
    std::map<Price, Qty> ref_bids, ref_asks;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> op_dist(0, 2);
    std::uniform_int_distribution<int> price_off_dist(0, static_cast<int>(NUM_LEVELS) - 1);
    std::uniform_int_distribution<int> qty_dist(1, 1000);
    OrderId next_id = 1;
    std::vector<OrderId> live_ids;

    auto ref_add = [&](Price price, Qty qty, Side side) {
        auto& m = (side == Side::Bid) ? ref_bids : ref_asks;
        m[price] += qty;
    };

    auto ref_remove = [&](Price price, Qty qty, Side side) {
        auto& m = (side == Side::Bid) ? ref_bids : ref_asks;
        auto it = m.find(price);
        CHECK(it != m.end());
        it->second -= qty;;
        if (it->second == 0) m.erase(it);
    };
    
    for (int op = 0; op < num_ops; ++op) {
        int kind = live_ids.empty() ? 0 : op_dist(rng);
        if (kind == 0) {
            Price price = BASE + price_off_dist(rng);
            Qty qty = qty_dist(rng);
            Side side = (rng() & 1) ? Side::Bid : Side::Ask;
            OrderId id = next_id++;
            CHECK(book->add_order(id, side, price, qty));
            ref[id] = RefOrder{side, price, qty};
            ref_add(price, qty, side);
            live_ids.push_back(id);
        } else if (kind == 1) {
            std::size_t idx = rng() % live_ids.size();
            OrderId id = live_ids[idx];
            RefOrder r = ref[id];
            CHECK(book->cancel_order(id));
            ref_remove(r.price, r.qty, r.side);
            ref.erase(id);
            live_ids[idx] = live_ids.back();
            live_ids.pop_back();
        } else {
            std::size_t idx = rng() % live_ids.size();
            OrderId id = live_ids[idx];
            RefOrder& r = ref[id];
            Qty new_qty = 1 + (rng() % r.qty);
            CHECK(book->modify_qty(id, new_qty));
            ref_remove(r.price, r.qty, r.side);
            r.qty = new_qty;
            ref_add(r.price, r.qty, r.side);
        }

        //cross-check best bid/ask against the ref after every op
        Price bp, ap;
        Qty bq, aq;
        bool has_bid = book->best_bid(bp, bq);
        bool has_ask = book->best_ask(ap, aq);
        CHECK(has_bid == !ref_bids.empty());
        CHECK(has_ask == !ref_asks.empty());
        if (has_bid) {
            auto it = ref_bids.rbegin();
            CHECK(bp == it->first && bq == it->second);
        }
        if (has_ask) {
            auto it = ref_asks.begin();
            CHECK(ap == it->first && aq == it->second);
        }
    }
    std::printf("test_random_differential(seed=%u, ops=%d): OK\n", seed, num_ops);
}

// Rough benchmark, not a proper performance test, just a sanity check that the code runs at all.

void bench_rough() {
    constexpr std::size_t NUM_LEVELS = 65536;
    constexpr std::size_t MAX_ORDERS = 200000;
    auto book = std::make_unique<OrderBook<NUM_LEVELS, MAX_ORDERS>>(/*base_price=*/1'000'000);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> price_off_dist(0, static_cast<int>(NUM_LEVELS) - 1);

    const int N = 200000;
    std::vector<OrderId> ids(N);
    for (int i = 0; i < N; ++i) ids[i] = i+1;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        book->add_order(ids[i], (i & 1) ? Side::Bid : Side::Ask, 1'000'000 + price_off_dist(rng), 100);
    }
    //not optimized
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        Price p = 0;
        Qty q = 0;
        book->best_bid(p, q);
        #ifdef __GNUC__
                // GCC/Clang on Linux will compile this
                asm volatile("" : : "r"(p), "r"(q) : "memory"); 
        #else
                // MSVC IntelliSense on Windows will fall back to this, keeping the editor happy
                std::atomic_signal_fence(std::memory_order_seq_cst);
        #endif
        book->best_ask(p, q);
        #ifdef __GNUC__
                asm volatile("" : : "r"(p), "r"(q) : "memory");
        #else
                std::atomic_signal_fence(std::memory_order_seq_cst);
        #endif
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        book->cancel_order(ids[i]);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    
    auto ns = [](auto d) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
    };
    
    std::printf("add: %6.1f ns/op\n", double(ns(t1-t0)) / N);
    std::printf("best b+a: %6.1f ns/op\n", double(ns(t2-t1)) / N);
    std::printf("cancel: %6.1f ns/op\n", double(ns(t3-t2)) / N);
    std::puts("(sandboxed/VM env, no core pinning or turbo control here" 
              " -- treat as a rough sanity check, not a proper benchmark)");
}

int main() {
    test_basic();
    test_random_differential(1, 50000);
    test_random_differential(2, 50000);
    test_random_differential(12345, 100000);
    bench_rough();
    std::puts("all tests passed");
    return 0;
}