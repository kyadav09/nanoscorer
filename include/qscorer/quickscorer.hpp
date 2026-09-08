#pragma once
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#include "tree.hpp"

namespace qs {

/**
 * qs: High-Performance Decision Tree Inference
 * 
 * Replaces standard branch-heavy tree traversal with bitwise mask elimination.
 * By removing `if/else` control flow, this architecture entirely bypasses CPU 
 * branch predictor penalties and guarantees constant-time execution latency 
 * regardless of tree depth or input data variance.
 */

struct QSNode {
    float threshold;
    uint32_t feature;
    uint64_t false_mask;
};

static_assert(sizeof(QSNode) == 16, "QSNode must be exactly 16 bytes for cache line packing, 4 nodes per cache line");

class QSTree {
    std::vector<QSNode> nodes_;
    std::vector<float> leaf_values_;

    //temp context struct passed during tree-flattening
    struct BuildCtx {
        std::vector<QSNode> nodes;
        std::vector<float> leaves;
    };

    static int count_leaves(const TreeNode* n) {
        if (n->is_leaf) return 1;
        return count_leaves(n->left.get()) + count_leaves(n->right.get());
    }

    //returns [first, last] inclusive leaf-bit range covered by this subtree from left to right as leaf bits are contiguous
    static std::pair<int, int> assign(const TreeNode* n, BuildCtx& ctx) {
        if (n->is_leaf) {
            int idx = static_cast<int>(ctx.leaves.size());
            ctx.leaves.push_back(n->value);
            return {idx, idx};
        }

        auto [lf, ll] = assign(n->left.get(), ctx);
        auto [rf, rl] = assign(n->right.get(), ctx);
        (void)rf;

        uint64_t clear_range;

        int width = ll - lf + 1;
        if (width >= 64) {
            clear_range = ~uint64_t{0};
        } else {
            clear_range = ((uint64_t{1} << width) - 1) << lf;
        }

        ctx.nodes.push_back(QSNode{
            n->threshold,
            static_cast<uint32_t>(n->feature),
            ~clear_range
        });

        return {lf, rl};
    }

public:
    static QSTree build(const TreeNode* root) {
        if (count_leaves(root) > 64) {
            throw std::runtime_error("QSTree::build: Tree exceeds 64 leaves");
        }
        
        BuildCtx ctx;
        assign(root, ctx);
        
        QSTree t;
        t.nodes_ = std::move(ctx.nodes);
        t.leaf_values_ = std::move(ctx.leaves);
        return t;
    }

    //Branchless execution - bitwise elimination of impossible leaves
    int leaf_index_branchless(const float* x) const {
        uint64_t mask = ~uint64_t{0};
        for (const QSNode& n: nodes_) {
            bool go_left = (x[n.feature] < n.threshold);
            uint64_t keep_all = -static_cast<uint64_t>(go_left);
            mask &= (n.false_mask | keep_all);
        }

        return std::countr_zero(mask);
    }

    float eval_branchless(const float* x) const {
        return leaf_values_[static_cast<std::size_t>(leaf_index_branchless(x))];
    }

    //exists to compare against eval_branchless specifically for benchmarking
    float eval_branchy(const float* x) const {
        uint64_t mask = ~uint64_t{0};
        for (const QSNode& n: nodes_) {
            if (!(x[n.feature] < n.threshold)) {
                mask &= n.false_mask;
            }
        }
        int leaf = std::countr_zero(mask);
        return leaf_values_[static_cast<std::size_t>(leaf)];
    }

    std::size_t num_nodes() const { return nodes_.size(); }
    std::size_t num_leaves() const { return leaf_values_.size(); }

    const std::vector<QSNode>& nodes() const { return nodes_; }
    const std::vector<float>& leaf_values() const { return leaf_values_; }

};

//sum of trees + base_score, forest
class GBDTEnsemble {
    std::vector<QSTree> trees_;
    double base_score_ = 0.0;

public: 
    void set_base_score(double b) { base_score_ = b; }
    void add_tree(QSTree t) { trees_.push_back(std::move(t)); }

    double predict(const float* x ) const {
        double sum = base_score_;
        for (const auto& t: trees_) sum += t.eval_branchless(x);
        return sum;
    }

    double predict_branchy(const float* x) const {
        double sum = base_score_;
        for (const auto& t: trees_) sum += t.eval_branchy(x);
        return sum;
    }

    std::size_t num_trees() const { return trees_.size(); }
};

} // namespace qs