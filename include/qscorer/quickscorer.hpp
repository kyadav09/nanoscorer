#pragma once
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#include "tree.hpp"

namespace qs {

// One

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
            throw std::runtime_error("QSTree::build: Tree exceeds 64 leaves")
        }
        
        BuildCtx ctx;
        assign(root, ctx);
        
        QSTree t;
        t.nodes_ = std::move(ctx.nodes);
        t.leaf_values_ = std::move(ctx.leaves);
        return t;
    }

    

};

}