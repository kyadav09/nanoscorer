#include "qscorer/quickscorer.hpp"
#include "qscorer/quickscorer_avx2.hpp"
#include "qscorer/tree.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

using namespace qs;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "CHECK FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            std::abort(); \
        } \
    } while (0) \

template <typename T>
inline void do_not_optimize(T const& value) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    volatile T sink = value;
    (void)sink;
#endif
}

static std::unique_ptr<TreeNode> random_tree(std::mt19937& rng, int depth, int max_depth, int num_features) {
    std::uniform_real_distribution<float> val_dist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> thresh_dist(-10.0f, 10.0f);
    std::uniform_int_distribution<int> feat_dist(0, num_features - 1);
    std::bernoulli_distribution stop_dist(0.3);

    if (depth >= max_depth || (depth > 0 && stop_dist(rng))) {
        return TreeNode::make_leaf(val_dist(rng));
    }
    int feature = feat_dist(rng);
    float threshold = thresh_dist(rng);
    auto left = random_tree(rng, depth + 1, max_depth, num_features);
    auto right = random_tree(rng, depth + 1, max_depth, num_features);
    return TreeNode::make_split(feature, threshold, std::move(left), std::move(right));
}

static std::unique_ptr<TreeNode> full_tree(int depth) {
    if (depth == 0) return TreeNode::make_leaf(0.0f);
    return TreeNode::make_split(0, 0.0f, full_tree(depth - 1), full_tree(depth - 1));
}

void test_basic() {
    // Exactly 32 leaves: the boundary case that must succeed.
    {
        auto root = full_tree(5); // 2^5 = 32 leaves
        assign_leaf_bits(root.get());
        QSTree qst = QSTree::build(root.get());
        CHECK(qst.num_leaves() == 32);
        QSTreeAVX2 avx = QSTreeAVX2::build(qst); // should not throw
        CHECK(avx.num_nodes() == qst.num_nodes());
    }
    
    // exceeds 32 limit -> must throw
    {
        auto root = full_tree(6); // 64 leaves
        assign_leaf_bits(root.get());
        QSTree qst = QSTree::build(root.get());
        bool threw = false;
        try {
            QSTreeAVX2::build(qst);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
    }
    
    //SoA batch test
    {
        auto root = TreeNode::make_split(
            0, 5.0f, TreeNode::make_leaf(1.0f),
            TreeNode::make_split(1, 3.0f, TreeNode::make_leaf(2.0f), TreeNode::make_leaf(3.0f)));
        assign_leaf_bits(root.get());
        QSTree qst = QSTree::build(root.get());
        QSTreeAVX2 avx = QSTreeAVX2::build(qst);

        // feature-major: xs[feature*8 + lane]
        float xs[2 * 8];
        float expected[8];
        const float f0[8] = {1, 9, 9, 1, 9, 9, 1, 9};
        const float f1[8] = {0, 1, 9, 0, 1, 9, 0, 1};
        const float exp[8] = {1, 2, 3, 1, 2, 3, 1, 2};
        for (int i = 0; i < 8; ++i) {
            xs[0 * 8 + i] = f0[i];
            xs[1 * 8 + i] = f1[i];
            expected[i] = exp[i];
        }
        float out[8];
        avx.eval_batch8(xs, out);
        for (int i = 0; i < 8; ++i) CHECK(out[i] == expected[i]);
    }
    std::puts("test_basic: OK");
}

void test_random_differential(uint32_t seed, int num_trees, int num_features, int max_depth, int batches_per_tree) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> x_dist(-10.0f, 10.0f);

    int trees_tested = 0;
    for (int t = 0; t < num_trees && trees_tested < num_trees; ++t) {
        auto root = random_tree(rng, 0, max_depth, num_features);
        assign_leaf_bits(root.get());
        QSTree qst = QSTree::build(root.get());
        if (qst.num_leaves() > 32) continue; // AVX2 form doesn't apply; skip, don't fudge it
        QSTreeAVX2 avx = QSTreeAVX2::build(qst);
        ++trees_tested;

        std::vector<float> xs(static_cast<size_t>(num_features) * 8);
        std::vector<std::vector<float>> rows(8, std::vector<float>(static_cast<size_t>(num_features)));

        for (int b = 0; b < batches_per_tree; ++b) {
            for (int lane = 0; lane < 8; ++lane) {
                for (int f = 0; f < num_features; ++f) {
                    float v = x_dist(rng);
                    rows[static_cast<size_t>(lane)][static_cast<size_t>(f)] = v;
                    xs[static_cast<size_t>(f) * 8 + static_cast<size_t>(lane)] = v;
                }
            }
            float out[8];
            avx.eval_batch8(xs.data(), out);
            for (int lane = 0; lane < 8; ++lane) {
                float scalar_result = qst.eval_branchless(rows[static_cast<size_t>(lane)].data());
                CHECK(out[lane] == scalar_result);
            }
        }
    }
    std::printf("test_random_differential(seed=%u, trees=%d, depth<=%d): OK (%d trees had <=32 leaves)\n", seed, num_trees, max_depth, trees_tested);
}

// Rough sanity benchmark -- same sandbox caveats as before (no pinning,
// shared single core). Comparing throughput of "8 separate scalar calls"
// against "one batch-of-8 AVX2 call" for the same tree and the same 8
// inputs each round; inputs still freshly randomized per round so nothing
// gets to memorize a pattern.
void bench_batch8() {
    constexpr int NUM_FEATURES = 16;
    std::mt19937 rng(555);
    auto root = random_tree(rng, 0, /*max_depth=*/5, NUM_FEATURES); // <=32 leaves, guaranteed
    assign_leaf_bits(root.get());
    QSTree qst = QSTree::build(root.get());
    QSTreeAVX2 avx = QSTreeAVX2::build(qst);

    constexpr int ROUNDS = 200000;
    std::uniform_real_distribution<float> x_dist(-10.0f, 10.0f);

    // Pre-generate ROUNDS batches, feature-major, so generation isn't
    // inside the timed region.
    std::vector<float> xs_batches(static_cast<size_t>(ROUNDS) * NUM_FEATURES * 8);
    for (auto& v : xs_batches) v = x_dist(rng);
    // Same data, re-laid-out row-major (per lane) for the scalar path, so
    // both versions score identical inputs each round.
    std::vector<float> rows_batches(static_cast<size_t>(ROUNDS) * 8 * NUM_FEATURES);
    for (int r = 0; r < ROUNDS; ++r) {
        const float* xs = &xs_batches[static_cast<size_t>(r) * NUM_FEATURES * 8];
        float* rows = &rows_batches[static_cast<size_t>(r) * 8 * NUM_FEATURES];
        for (int lane = 0; lane < 8; ++lane)
            for (int f = 0; f < NUM_FEATURES; ++f) rows[lane * NUM_FEATURES + f] = xs[f * 8 + lane];
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < ROUNDS; ++r) {
        float out[8];
        avx.eval_batch8(&xs_batches[static_cast<std::size_t>(r) * NUM_FEATURES * 8], out);
        do_not_optimize(out[0]);
        do_not_optimize(out[7]);
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int r = 0; r < ROUNDS; ++r) {
        const float* rows = &rows_batches[static_cast<std::size_t>(r) * 8 * NUM_FEATURES];
        float acc = 0;
        for (int lane = 0; lane < 8; ++lane) acc += qst.eval_branchless(rows + lane * NUM_FEATURES);
        do_not_optimize(acc);
    }
    auto t2 = std::chrono::steady_clock::now();

    auto ns_per_round = [](auto d) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count() / double(ROUNDS);
    };
    double avx_ns = ns_per_round(t1 - t0);
    double scalar_ns = ns_per_round(t2 - t1);
    std::printf("avx2 batch8: %6.1f ns/round  (%.2f ns/instance)\n", avx_ns, avx_ns / 8.0);
    std::printf("scalar x8 loop:  %6.1f ns/round  (%.2f ns/instance)\n", scalar_ns, scalar_ns / 8.0);
    std::puts("(sandboxed/VM environment, single shared core, no pinning -- rough, not final)");
}

int main() {
    test_basic();
    test_random_differential(1, /*num_trees=*/300, /*num_features=*/8, /*max_depth=*/5, /*batches_per_tree=*/100);
    test_random_differential(2, 300, 4, 4, 100);
    bench_batch8();
    std::puts("all tests passed");
    return 0;
}
