#pragma once
#if !defined(__AVX2__)
#error "quickscorer_avx2.hpp needs -mavx2. Deliberately not folded into the \
default build flags -- see README for why blanket -march=native stops \
being safe once real intrinsics are in the tree."
#endif
#include <bit>
#include <cassert>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>
#include <vector>
#include "quickscorer.hpp"

namespace qs {

    /**
     * AVX2-accelerated QuickScorer GBDT inference engine.
     * 
     * Maps the branchless bitwise-elimination algorithm directly onto 256-bit YMM 
     * registers to evaluate 8 independent market states simultaneously. Strictly 
     * expects a Structure of Arrays (SoA) memory layout to guarantee contiguous, 
     * zero-stride SIMD memory loads for deterministic low-latency execution.
     */

    //Evals 8 instances concurrently using AVX2
    // Max 32 leaves so 8 lanes of 32-bit masks fit into one 256-bit register
class QSTreeAVX2 {
    struct Node32 {
        float threshold;
        uint32_t feature;
        uint32_t false_mask;
    };

    std::vector<Node32> nodes_;
    std::vector<float> leaf_values_;

public:
    static QSTreeAVX2 build(const QSTree& src) {
        if (src.num_leaves() > 32) {
            throw std::runtime_error (
                "QSTreeAVX2::build: tree has more than 32 leaves; 8 lanes of "
                "64-bit masks don't fit in one __m256i and thsi implementation "
                "doesn't split a mask across two registers. "
            );
        }
        QSTreeAVX2 t;
        t.nodes_.reserve(src.nodes().size());
        for (const QSNode& n: src.nodes()) {
            t.nodes_.push_back(Node32{n.threshold, n.feature, static_cast<uint32_t>(n.false_mask)});
        }
        t.leaf_values_ = src.leaf_values();
        return t;
    }

    //Expects SoA layout: xs_feature_major[feature * 8 + lane]
    // Allows single continguous SIMD loads per feature, preventing slow strided gathers
    void eval_batch8(const float* xs_feature_major, float* out) const {
        __m256i mask = _mm256_set1_epi32(-1);
        for (const Node32& n: nodes_) {
            const float* col = xs_feature_major + static_cast<std::size_t>(n.feature) * 8;
            __m256 xvals = _mm256_loadu_ps(col);
            __m256 thresh = _mm256_set1_ps(n.threshold);

            // hardware compares outputs 0xFFFFFFF if x < thresh (go left) else 0
            __m256i go_left = _mm256_castps_si256(_mm256_cmp_ps(xvals, thresh, _CMP_LT_OQ));
            __m256i false_mask_vec = _mm256_set1_epi32(static_cast<int32_t>(n.false_mask));
            
            // bitwise path elim
            mask = _mm256_and_si256(mask, _mm256_or_si256(false_mask_vec, go_left));
        }
        alignas(32) uint32_t lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), mask);
        for (int i = 0; i < 8; ++i) {
            out[i] = leaf_values_[static_cast<std::size_t>(std::countr_zero(lanes[i]))];
        }
    }

    std::size_t num_nodes() const { return nodes_.size(); }
};

class GBDTEnsembleAVX2 {
    std::vector<QSTreeAVX2> trees_;
    double base_score_ = 0.0;

public: 
    void set_base_score(double b) { base_score_ = b; }
    void add_tree(QSTreeAVX2 t) { trees_.push_back(std::move(t)); }

    void predict_batch8(const float* xs_feature_major, float* out) const {
        for (int i = 0; i < 8; ++i) {
            out[i] = static_cast<float>(base_score_);   
        }
        alignas(32) float tree_out[8];
        for (const auto& t : trees_) {
            t.eval_batch8(xs_feature_major, tree_out);
            for (int i = 0 ; i < 8; ++i) {
                out[i] += tree_out[i];
            }
        }
    }

    std::size_t num_trees() const { return trees_.size(); }
};

}