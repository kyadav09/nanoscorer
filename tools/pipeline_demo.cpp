// simply a demonstration artifact / standalone tool, not for other code to link against
// integration step, feed order -> ring buffer -> inference thread: updates real order book
// , derives features from it, and scores from those features with real QuickScorer ensemble
// this file makes sure tests compose correctly, end to end hot path latency for components
// to properly report numbers
#include "orderbook/order_book.hpp"
#include "pipeline/order_event.hpp"
#include "pipeline/spsc_ring_buffer.hpp"
#include "qscorer/quickscorer.hpp"
#include "qscorer/tree.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

using namespace ob;
using namespace qs;
using namespace pipeline;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "CHECK FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            std::abort(); \
        } \
    } while(0) \

constexpr int NUM_FEATURES = 6;

// Feature definitions: [bid_offset, ask_offset, spread, bid_qty, ask_qty, imbalance]
struct FeatureRange {
    float lo, hi;
};
constexpr FeatureRange FEATURE_RANGES[NUM_FEATURES] = {
    {0.0f, 8192.0f},  // 0: best bid price offset
    {0.0f, 8192.0f},  // 1: best ask price offset
    {0.0f, 200.0f},   // 2: spread
    {0.0f, 1000.0f},  // 3: best bid quantity
    {0.0f, 1000.0f},  // 4: best ask quantity
    {-1.0f, 1.0f},    // 5: imbalance ratio
};

// Snapshots top-of-book state to derive a 6-factor quantitative feature vector.
template <size_t NUM_LEVELS, std::size_t MAX_ORDERS>
bool extract_features(const OrderBook<NUM_LEVELS, MAX_ORDERS>& book, Price base, float out[NUM_FEATURES]) {
    Price bp = 0, ap = 0;
    Qty bq = 0, aq = 0;
    bool has_bid = book.best_bid(bp, bq);
    bool has_ask = book.best_ask(ap, aq);
    if (!has_bid || !has_ask) return false;
    out[0] = static_cast<float>(bp - base);
    out[1] = static_cast<float>(ap - base);
    out[2] = static_cast<float>(ap - bp);
    out[3] = static_cast<float>(bq);
    out[4] = static_cast<float>(aq);
    out[5] = static_cast<float>(bq - aq) / static_cast<float>(bq + aq);
    return true;
}

// Generates a mock decision tree with thresholds bound to realistic feature ranges.
static std::unique_ptr<TreeNode> random_tree(std::mt19937& rng, int depth, int max_depth) {
    std::uniform_int_distribution<int> feat_dist(0, NUM_FEATURES - 1);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::bernoulli_distribution stop_dist(0.3);
    if (depth >= max_depth || (depth > 0 && stop_dist(rng))) {
        return TreeNode::make_leaf(val_dist(rng));
    }
    int feature = feat_dist(rng);
    std::uniform_real_distribution<float> thresh_dist(FEATURE_RANGES[feature].lo, FEATURE_RANGES[feature].hi);
    float threshold = thresh_dist(rng);
    auto left = random_tree(rng, depth + 1, max_depth);
    auto right = random_tree(rng, depth + 1, max_depth);
    return TreeNode::make_split(feature, threshold, std::move(left), std::move(right));
}

// Builds a synthetic GBDT forest for load-testing the inference engine.
GBDTEnsemble build_demo_ensemble(uint32_t seed, int num_trees, int max_depth) {
    std::mt19937 rng(seed);
    GBDTEnsemble ensemble;
    ensemble.set_base_score(0.0);
    for (int t = 0; t < num_trees; ++t) {
        auto root = random_tree(rng, 0, max_depth);
        assign_leaf_bits(root.get());
        ensemble.add_tree(QSTree::build(root.get()));
    }
    return ensemble;
}

struct LiveOrder {
    OrderId id;
    Qty qty;
};

// Emits a randomized but structurally valid stream of Add/Cancel/Modify order events.
std::vector<OrderEvent> generate_events(uint32_t seed, int count, int num_levels, Price base) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> op_dist(0, 2);
    std::uniform_int_distribution<int> price_off(0, num_levels - 1);
    std::uniform_int_distribution<Qty> qty_dist(1, 1000);
    std::vector<OrderEvent> events;
    events.reserve(static_cast<std::size_t>(count));
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
            std::size_t idx = rng() % live.size();
            e.type = EventType::Cancel;
            e.id = live[idx].id;
            live[idx] = live.back();
            live.pop_back();
        } else {
            std::size_t idx = rng() % live.size();
            e.type = EventType::Modify;
            e.id = live[idx].id;
            e.qty = 1 + (rng() % live[idx].qty);
            live[idx].qty = e.qty;
        }
        events.push_back(e);
    }
    return events;
}

template <std::size_t NUM_LEVELS, std::size_t MAX_ORDERS>
void apply_event(OrderBook<NUM_LEVELS, MAX_ORDERS>& book, const OrderEvent& e) {
    switch (e.type) {
        case EventType::Add: book.add_order(e.id, e.side, e.price, e.qty); break;
        case EventType::Cancel: book.cancel_order(e.id); break;
        case EventType::Modify: book.modify_qty(e.id, e.qty); break;
    }
}

// Computes tail latencies (p50, p90, p99, p99.9) typical of HFT performance analysis.
void print_percentiles(std::vector<long long>& ns, const char* label) {
    std::sort(ns.begin(), ns.end());
    auto pct = [&](double p) { return ns[static_cast<std::size_t>(p * double(ns.size() - 1))]; };
    std::printf("%s: n=%zu  p50=%lld ns  p90=%lld ns  p99=%lld ns  p99.9=%lld ns  max=%lld ns\n", label, ns.size(), pct(0.50), pct(0.90), pct(0.99), pct(0.999), ns.back());
}

int main() {
    constexpr std::size_t NUM_LEVELS = 8192;
    constexpr std::size_t MAX_ORDERS = 200000;
    constexpr Price BASE = 0;
    constexpr int NUM_EVENTS = 200000;
    
    auto events = generate_events(/*seed=*/42, NUM_EVENTS, static_cast<int>(NUM_LEVELS), BASE);
    GBDTEnsemble ensemble = build_demo_ensemble(/*seed=*/7, /*num_trees=*/50, /*max_depth=*/5);

    auto ref_book = std::make_unique<OrderBook<NUM_LEVELS, MAX_ORDERS>>(BASE);
    std::vector<double> ref_predications;
    for (const auto& e: events) {
        apply_event(*ref_book, e);
        float feat[NUM_FEATURES];
        if (extract_features(*ref_book, BASE, feat))
            ref_predications.push_back(ensemble.predict(feat));
    }

    auto live_book = std::make_unique<OrderBook<NUM_LEVELS, MAX_ORDERS>>(BASE);
    SpscRingBuffer<OrderEvent, 4096> rb;
    std::atomic<bool> producer_done{false};
    std::vector<double> live_predictions;
    live_predictions.reserve(events.size());
    std::vector<long long> latencies_ns;
    latencies_ns.reserve(events.size());
    std::thread producer([&] {
        for (const auto& e: events) 
            while (!rb.try_push(e))
                std::this_thread::yield();
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&] {
        OrderEvent e;
        auto process_one = [&] {
            auto t0 = std::chrono::steady_clock::now();
            apply_event(*live_book, e);
            float feat[NUM_FEATURES];
            double pred = 0.0;
            bool have_pred = extract_features(*live_book, BASE, feat);
            if (have_pred) pred = ensemble.predict(feat);
            auto t1 = std::chrono::steady_clock::now();
            if (have_pred) {
                live_predictions.push_back(pred);
                latencies_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            }
        };
    });

    return 0;
}


