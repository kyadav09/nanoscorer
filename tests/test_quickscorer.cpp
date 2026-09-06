#include "qscorer/quickscorer.hpp"
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

void test_basic() {
    // Single-leaf tree: no internal nodes at all.
    {
        auto root = TreeNode::make_leaf(42.0f);
        assign_leaf_bits(root.get());
        QSTree qst = QSTree::build(root.get());
        CHECK(qst.num_nodes() == 0 && qst.num_leaves() == 1);
        float x[1] = {0.0f};
        CHECK(qst.eval_branchless(x) == 42.0f);
        CHECK(qst.eval_branchy(x) == 42.0f);
    }

    // 3-leaf tree
    {
        auto root = TreeNode::make_split(
            0, 5.0f, TreeNode::make_leaf(1.0f),
            TreeNode::make_split(1, 3.0f, TreeNode::make_leaf(2.0f), TreeNode::make_leaf(3.0f)));
        assign_leaf_bits(root.get());
        QSTree qst = QSTree::build(root.get());
        CHECK(qst.num_nodes() == 2 && qst.num_leaves() == 3);

        float xA[2] = {1.0f, 0.0f};  // A
        float xB[2] = {9.0f, 1.0f};  // B
        float xC[2] = {9.0f, 9.0f};  // C
        CHECK(qst.eval_branchless(xA) == 1.0f && qst.eval_branchy(xA) == 1.0f);
        CHECK(qst.eval_branchless(xB) == 2.0f && qst.eval_branchy(xB) == 2.0f);
        CHECK(qst.eval_branchless(xC) == 3.0f && qst.eval_branchy(xC) == 3.0f);
    }
   
    //Exceeds 64 leaves limit
    {
        std::mt19937 rng(999);
        std::function<std::unique_ptr<TreeNode>(int)> full = [&](int depth) -> std::unique_ptr<TreeNode> {
            if (depth == 0) return TreeNode::make_leaf(0.0f);
            return TreeNode::make_split(0, 0.0f, full(depth - 1), full(depth - 1));
        };
        auto big_root = full(7); // 128 leaves
        assign_leaf_bits(big_root.get());
        bool threw = false;
        try {
            QSTree::build(big_root.get());
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
    }
    std::puts("test_basic: OK");
}

void test_random_differential(uint32_t seed, int num_trees, int num_features, int max_depth, int inputs_per_tree) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> x_dist(-10.0f, 10.0f);
    std::vector<float> x(static_cast<size_t>(num_features));

    for (int t = 0; t < num_trees; ++t) {
        auto root = random_tree(rng, 0, max_depth, num_features);
        assign_leaf_bits(root.get());
        QSTree qst = QSTree::build(root.get());

        for (int i = 0; i < inputs_per_tree; ++i) {
            for (auto& v : x) v = x_dist(rng);

            const TreeNode* reached = naive_walk(root.get(), x.data());

            CHECK(qst.leaf_index_branchless(x.data()) == reached->leaf_bit);
            CHECK(qst.eval_branchless(x.data()) == reached->value);
            CHECK(qst.eval_branchy(x.data()) == reached->value);
        }
    }
    std::printf("test_random_differential(seed=%u, trees=%d, depth<=%d): OK\n", seed, num_trees, max_depth);
}

void test_ensemble_differential(uint32_t seed, int num_ensembles, int trees_per_ensemble, int num_features, int max_depth, int inputs_per_ensemble) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> x_dist(-10.0f, 10.0f);
    std::uniform_real_distribution<double> base_dist(-5.0, 5.0);
    std::vector<float> x(static_cast<size_t>(num_features));

    for (int e = 0; e < num_ensembles; ++e) {
        double base = base_dist(rng);
        GBDTEnsemble ensemble;
        ensemble.set_base_score(base);
        std::vector<std::unique_ptr<TreeNode>> roots;
        for (int t = 0; t < trees_per_ensemble; ++t) {
            auto root = random_tree(rng, 0, max_depth, num_features);
            assign_leaf_bits(root.get());
            ensemble.add_tree(QSTree::build(root.get()));
            roots.push_back(std::move(root));
        }

        for (int i = 0; i < inputs_per_ensemble; ++i) {
            for (auto& v : x) v = x_dist(rng);

            double expected = base;
            for (const auto& root : roots) expected += naive_walk(root.get(), x.data())->value;

            CHECK(ensemble.predict(x.data()) == expected);
            CHECK(ensemble.predict_branchy(x.data()) == expected);
        }
    }
    std::printf("test_ensemble_differential(seed=%u, ensembles=%d, trees=%d): OK\n", seed, num_ensembles, trees_per_ensemble);
}

// Rough sanity benchmark -- same caveats as the order book's: no core
// pinning or turbo control in this sandbox. The point of this one isn't
// the absolute numbers, it's branchless vs. branchy on the *same* tree.
// Inputs are freshly randomized per call and pre-generated into a flat
// buffer (not reused, not regenerated in the timed region) specifically
// so the branch predictor can't learn a fixed pattern and hide the effect
// this comparison exists to measure.
void bench_branchless_vs_branchy() {
    constexpr int NUM_FEATURES = 16;
    std::mt19937 rng(777);
    auto root = random_tree(rng, 0, /*max_depth=*/6, NUM_FEATURES);
    QSTree qst = QSTree::build(root.get());

    constexpr int N = 500000;
    std::uniform_real_distribution<float> x_dist(-10.0f, 10.0f);
    std::vector<float> inputs(static_cast<size_t>(N) * NUM_FEATURES);
    for (auto& v : inputs) v = x_dist(rng);

    auto time_it = [&](auto&& fn) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            float r = fn(&inputs[static_cast<size_t>(i) * NUM_FEATURES]);
            do_not_optimize(r);
        }
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / double(N);
    };

    double branchless_ns = time_it([&](const float* x) { return qst.eval_branchless(x); });
    double branchy_ns = time_it([&](const float* x) { return qst.eval_branchy(x); });

    std::printf("branchless: %6.2f ns/eval  (%zu nodes, %zu leaves)\n", branchless_ns, qst.num_nodes(), qst.num_leaves());
    std::printf("branchy: %6.2f ns/eval\n", branchy_ns);
    std::puts("(sandboxed/VM environment, single shared core, no pinning -- rough, not final)");
}

int main() {
    test_basic();
    test_random_differential(1, /*num_trees=*/300, /*num_features=*/8, /*max_depth=*/6, /*inputs_per_tree=*/300);
    test_random_differential(2, 300, 4, 3, 300);
    test_ensemble_differential(3, /*num_ensembles=*/50, /*trees_per_ensemble=*/40, /*num_features=*/8, /*max_depth=*/5, /*inputs_per_ensemble=*/50);
    bench_branchless_vs_branchy();
    std::puts("all tests passed");
    return 0;
}
