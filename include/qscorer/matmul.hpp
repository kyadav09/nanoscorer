#pragma once
#include <algorithm>
#include <cstddef>

// Square n x n row-major float matmul implementation for micro-architectural benchmarking
// Computes C (overwritten) = A * B
namespace mm {

// standard ijk order / unoptimized baseline -> cachce line throttling
// stride = n
inline void naive_ijk(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// i-k-j order
// A[i][k] into scalar register (dot product)
// Inner loop with B and C: stride -= 1
// maximizes spactial locality and enables hardware prefetching and auto-vectorization
inline void naive_ikj(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i * n + j] = 0.0f;
    
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_ik = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

// Cache aware L1/L2 blocking
//  block_size is parameterized for empirical tuning
// Sub-block iteration order retains monotic k-traversal for exact bit-exact sum equality (naive)
inline void tiled(const float* A, const float* B, float* C, int n, int block_size) {
    for (int i = 0; i < n; ++i) 
        for (int j = 0; j < n; ++j)
            C[i * n + j] = 0.0f;

    for (int ii = 0; ii < n; ii += block_size) {
        int i_end = std::min(ii + block_size, n);
        for (int kk = 0; kk < n; kk += block_size) {
            int k_end = std::min(kk + block_size, n);
            for (int jj = 0; jj < n; jj += block_size) {
                int j_end = std::min(jj + block_size, n);
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        float a_ik = A[i * n + k];
                        for (int j = jj; j < j_end; ++j) {
                            C[i * n + j] += a_ik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

namespace detail {

// Base case size to amortize recursion overhead
// Very much hardware-independent (not tuned for specific cache size(s))
constexpr int COBLIVIOUS_BASE = 32;

// Computes sub-blocks using zero-copy strided views (lda ldb ldc) into original matrices
inline void base_multiply(const float* A, const float* B, float* C, int m, int k, int n, int lda, int ldb, int ldc, bool add_to_c) {
    if (!add_to_c) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; j++)
                C[i * ldc + j] = 0.0f;
    }
    for (int i = 0; i < m; ++i) {
        for (int kk = 0; kk < k; ++kk) {
            float a_ik = A[i * lda + kk];
            for (int j = 0; j < n; ++j) {
                C[i * ldc + j] += a_ik * B[kk * ldb + j];
            }
        }
    }
}

// Cache-oblivious dimension spliting
// Bisects largest dimension, recursively scaling working sets to fit L3, L2, L1 cache(s)
// caches implicitly w/o requiring hardware-specific tuning parameters
inline void recurse(const float* A, const float* B, float* C, int m, int k, int n, int lda, int ldb, int ldc, bool add_to_c) {
    if (m <= COBLIVIOUS_BASE && k <= COBLIVIOUS_BASE && n <= COBLIVIOUS_BASE) {
        base_multiply(A, B, C, m, k, n, lda, ldb, ldc, add_to_c);
        return;
    }
    if (m >= k && m >= n) {
        int m1 = m / 2, m2 = m - m1;
        recurse(A, B, C, m1, k, n, lda, ldb, ldc, add_to_c);
        recurse(A + static_cast<std::size_t>(m1) * lda, B, C + static_cast<std::size_t>(m1) * ldc, m2, k, n, lda, ldb, ldc, add_to_c);
    } else if (n >= m && n >= k) {
        int n1 = n / 2, n2 = n - n1;
        recurse(A, B, C, m, k, n1, lda, ldb, ldc, add_to_c);
        recurse(A, B + n1, C + n1, m, k, n2, lda, ldb, ldc, add_to_c);
    } else {
        // k-split: both halves target same region in C. 2nd half must alwyas accumulate (add_to_c = true)
                 // to preserve partial sums written by 1st half
        int k1 = k / 2, k2 = k - k1;
        recurse(A, B, C, m, k1, n, lda, ldb, ldc, add_to_c);
        recurse(A + k1, B + static_cast<std::size_t>(k1) * ldb, C, m, k2, n, lda, ldb, ldc, /*add_to_c=*/true);
    }
}

} // namespace detail

inline void cache_oblivious(const float* A, const float* B, float* C, int n) {
    detail::recurse(A, B, C, n, n, n, n, n, n, /*add_to_c=*/false);
}

} // namespace mm