#pragma once
#include <memory>

namespace qs {
// ordinary pter-based binary tree, walked node by node
// exists so Quick Scorer's flattened, branchless form has a reference implementation to compare against
// std::map ref for Order Book

struct TreeNode {
    bool is_leaf = false;

    //internal node data
    int feature = 0;
    float threshold = 0.0f;
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;

    //leaf node data
    float value = 0.0f;
    int leaf_bit = -1; // assing_leaf_bits() assigns this, so -1 for now

    static std::unique_ptr<TreeNode> make_leaf(float value) {
        auto n = std::make_unique<TreeNode>();
        n->is_leaf = true;
        n->value = value;
        return n;
    }
    static std::unique_ptr<TreeNode> make_split(int feature, float threshold, std::unique_ptr<TreeNode> l, std::unique_ptr<TreeNode> r) {
        auto n = std::make_unique<TreeNode>();
        n->is_leaf = false;
        n->feature = feature;
        n->threshold = threshold;
        n->left = std::move(l);
        n->right = std::move(r);
        return n;
    }
};

//basic branch walking
inline const TreeNode* naive_walk(const TreeNode* n, const float* x) {
    while (!n->is_leaf) {
        n = (x[n->feature] < n->threshold) ? n->left.get() : n->right.get();
    }
    return n;
}

//assigns leaf_bit in l-2-r order, same as QSTree::build uses
// we can dir compare leaf_bit instead of final value (which could match even if wrong leaf was chosen)
inline int assign_leaf_bits(TreeNode* n, int next = 0) {
    if (n->is_leaf) {
        n->leaf_bit = next;
        return next +1;
    }
    next = assign_leaf_bits(n->left.get(), next);
    next = assign_leaf_bits(n->right.get(), next);
    return next;
}

}