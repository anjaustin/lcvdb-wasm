#include <stdint.h>
#include <math.h>
#include <string.h>
#include "platform.h"
#include "lcvdb_ternary.h"

// Distance kernels for LCVDB Ternary (AVX2 / WASM SIMD / scalar)

// --- Ternary-2 Kernels (Scalar Fallbacks) ---

int32_t lcvdbt_dot_ternary(const uint8_t *a, const uint8_t *b, int loops) {
    int32_t dot = 0;
    for (int i = 0; i < loops * 16; i++) {
        uint8_t ba = a[i], bb = b[i];
        for (int j = 0; j < 4; j++) {
            int ta = (ba >> (j*2)) & 0x03;
            int tb = (bb >> (j*2)) & 0x03;
            int va = (ta == 1) ? 1 : (ta == 3) ? -1 : 0;
            int vb = (tb == 1) ? 1 : (tb == 3) ? -1 : 0;
            dot += va * vb;
        }
    }
    return dot;
}

int32_t lcvdbt_dot_ternary_ext(const uint8_t *a, const uint8_t *b, int32_t *support, int loops) {
    *support = 0;
    int32_t dot = 0;
    for (int i = 0; i < loops * 16; i++) {
        uint8_t ba = a[i], bb = b[i];
        for (int j = 0; j < 4; j++) {
            int ta = (ba >> (j*2)) & 0x03;
            int tb = (bb >> (j*2)) & 0x03;
            if (ta && tb) (*support)++;
            int va = (ta == 1) ? 1 : (ta == 3) ? -1 : 0;
            int vb = (tb == 1) ? 1 : (tb == 3) ? -1 : 0;
            dot += va * vb;
        }
    }
    return dot;
}

int32_t lcvdbt_l2sq_ternary(const uint8_t *a, const uint8_t *b, int loops) {
    int32_t l2 = 0;
    for (int i = 0; i < loops * 16; i++) {
        uint8_t ba = a[i], bb = b[i];
        for (int j = 0; j < 4; j++) {
            int ta = (ba >> (j*2)) & 0x03;
            int tb = (bb >> (j*2)) & 0x03;
            int va = (ta == 1) ? 1 : (ta == 3) ? -1 : 0;
            int vb = (tb == 1) ? 1 : (tb == 3) ? -1 : 0;
            int d = va - vb;
            l2 += d * d;
        }
    }
    return l2;
}

int32_t lcvdbt_dot_split(const uint8_t *a, const uint8_t *b, int loops) {
    int plane = loops * 16;
    const uint8_t *na = a, *sa = a + plane;
    const uint8_t *nb = b, *sb = b + plane;
    int32_t dot = 0;
    for (int i = 0; i < plane; i++) {
        uint8_t mask = na[i] & nb[i];
        uint8_t diff = sa[i] ^ sb[i];
        dot += __builtin_popcount(mask & ~diff);
        dot -= __builtin_popcount(mask & diff);
    }
    return dot;
}

int32_t lcvdbt_dot_split_ext(const uint8_t *a, const uint8_t *b, int32_t *support, int loops) {
    int plane = loops * 16;
    const uint8_t *na = a, *sa = a + plane;
    const uint8_t *nb = b, *sb = b + plane;
    int32_t dot = 0;
    *support = 0;
    for (int i = 0; i < plane; i++) {
        uint8_t mask = na[i] & nb[i];
        *support += __builtin_popcount(mask);
        uint8_t diff = sa[i] ^ sb[i];
        dot += __builtin_popcount(mask & ~diff);
        dot -= __builtin_popcount(mask & diff);
    }
    return dot;
}

int32_t lcvdbt_l2sq_split(const uint8_t *a, const uint8_t *b, int loops) {
    int plane = loops * 16;
    const uint8_t *na = a, *sa = a + plane;
    const uint8_t *nb = b, *sb = b + plane;
    int32_t l2 = 0;
    for (int i = 0; i < plane; i++) {
        uint8_t combined_nz = na[i] | nb[i];
        uint8_t common_nz = na[i] & nb[i];
        uint8_t diff_sign = sa[i] ^ sb[i];
        l2 += __builtin_popcount(combined_nz ^ common_nz); // One is zero, other is 1 -> dist 1
        l2 += 4 * __builtin_popcount(common_nz & diff_sign); // Both non-zero, diff sign -> dist 2 -> sq 4
    }
    return l2;
}

float lcvdbt_dot_f32_avx2(const float *a, const float *b, int loops) {
#ifdef LCVDB_AVX2
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < loops / 2; i++) {
        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(a + i*8), _mm256_loadu_ps(b + i*8)));
    }
    float f[8]; _mm256_storeu_ps(f, sum);
    return f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7];
#else
    float sum = 0;
    for (int i = 0; i < loops * 16; i++) sum += a[i] * b[i];
    return sum;
#endif
}

// --- Ternary-4 (int8) Kernels ---

int64_t lcvdbt_dot_ternary4(const int8_t *a, const int8_t *b, int dim) {
    int64_t total = 0;
    for (int i = 0; i < dim; i++) total += (int64_t)a[i] * b[i];
    return total;
}

int32_t lcvdbt_l2sq_ternary4(const int8_t *a, const int8_t *b, int dim) {
    int64_t total = 0;
    for (int i = 0; i < dim; i++) { int32_t d = (int32_t)a[i] - b[i]; total += (int64_t)d * d; }
    return (int32_t)total;
}

// --- Ternary-7 (int16) Kernels ---

#ifdef LCVDB_AVX2
int64_t lcvdbt_dot_ternary7_avx2(const int16_t *a, const int16_t *b, int loops) {
    __m256i sum0 = _mm256_setzero_si256();
    for (int i = 0; i < loops; i++) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i * 16));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i * 16));
        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(va, vb));
    }
    __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(sum0), _mm256_extracti128_si256(sum0, 1));
    s128 = _mm_hadd_epi32(s128, s128); s128 = _mm_hadd_epi32(s128, s128);
    return (int64_t)_mm_cvtsi128_si32(s128);
}

int64_t lcvdbt_dot_ternary7(const int16_t *a, const int16_t *b, int dim) {
    int loops = dim / 16;
    int64_t sum = lcvdbt_dot_ternary7_avx2(a, b, loops);
    for (int i = loops * 16; i < dim; i++) sum += (int64_t)a[i] * b[i];
    return sum;
}

int64_t lcvdbt_l2sq_ternary7(const int16_t *a, const int16_t *b, int dim) {
    int64_t sum = 0; int loops = dim / 16;
    if (loops > 0) {
        __m256i vsum = _mm256_setzero_si256();
        for (int i = 0; i < loops; i++) {
            __m256i va = _mm256_loadu_si256((const __m256i*)(a + i * 16));
            __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i * 16));
            __m256i vd = _mm256_sub_epi16(va, vb);
            vsum = _mm256_add_epi32(vsum, _mm256_madd_epi16(vd, vd));
        }
        __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(vsum), _mm256_extracti128_si256(vsum, 1));
        s128 = _mm_hadd_epi32(s128, s128); s128 = _mm_hadd_epi32(s128, s128);
        sum = (int64_t)_mm_cvtsi128_si32(s128);
    }
    for (int i = loops * 16; i < dim; i++) { int32_t d = (int32_t)a[i] - b[i]; sum += (int64_t)d * d; }
    return sum;
}

#endif /* LCVDB_AVX2 — ternary7 AVX2 + L2sq AVX2 */

// --- BPMT7 (True Ternary Compute) ---

// Scalar reference implementation (always available)
int64_t lcvdbt_dot_bpmt7_scalar(const uint8_t *a, const uint8_t *b, int dim) {
    int words = dim / 64; int pb = dim / 8; int64_t total = 0;
    static const int64_t w[] = {1,3,9,27,81,243,729,2187,6561,19683,59049,177147,531441};
    const uint64_t *pa = (const uint64_t *)a, *na = (const uint64_t *)(a + 7*pb);
    const uint64_t *pb_v = (const uint64_t *)b, *nb_v = (const uint64_t *)(b + 7*pb);
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            int64_t weight = w[i + j]; int64_t ij_sum = 0;
            for (int k = 0; k < words; k++) {
                uint64_t p_m = (pa[i*words+k] & pb_v[j*words+k]) | (na[i*words+k] & nb_v[j*words+k]);
                uint64_t n_m = (pa[i*words+k] & nb_v[j*words+k]) | (na[i*words+k] & pb_v[j*words+k]);
                ij_sum += (int64_t)__builtin_popcountll(p_m) - (int64_t)__builtin_popcountll(n_m);
            }
            total += weight * ij_sum;
        }
    }
    return total;
}

// --- AVX2 BPMT7 (fast dot product via popcount) ---
#ifdef LCVDB_AVX2

static inline __m256i _avx2_popcount256(__m256i v) {
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i lo_mask = _mm256_set1_epi8(0x0F);
    __m256i lo = _mm256_and_si256(v, lo_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lo_mask);
    __m256i cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                                   _mm256_shuffle_epi8(lut, hi));
    // vpsadbw against zero: sums each group of 8 bytes into a 64-bit lane
    return _mm256_sad_epu8(cnt, _mm256_setzero_si256());
}

// Accumulate popcount of v into running 64-bit sum register
static inline __m256i _avx2_popcnt_accum(__m256i acc, __m256i v) {
    return _mm256_add_epi64(acc, _avx2_popcount256(v));
}

// Horizontal sum of 4x 64-bit lanes in a __m256i
static inline int64_t _avx2_hsum_epi64(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi64(lo, hi);
    return _mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1);
}

// --- AVX2 BPMT7 Dot Product ---
//
// Processes 256 bits (4 x 64-bit words) per iteration instead of 64.
// For dim=768: words=12, chunks=3 (256-bit iterations per plane pair).
// For dim=384: words=6, chunks>=1 with scalar tail.
//
// Inner loop per plane pair (i,j):
//   p_m = (pa_i & pb_j) | (na_i & nb_j)   -- agree bits
//   n_m = (pa_i & nb_j) | (na_i & pb_j)   -- disagree bits
//   ij_sum += popcount(p_m) - popcount(n_m)

int64_t lcvdbt_dot_bpmt7(const uint8_t *a, const uint8_t *b, int dim) {
    const int words = dim / 64;
    const int pb = dim / 8;
    const int chunks = words / 4;   // 256-bit chunks
    int64_t total = 0;

    static const int64_t w[] = {1,3,9,27,81,243,729,2187,6561,19683,59049,177147,531441};

    const uint64_t *pa = (const uint64_t *)a;
    const uint64_t *na = (const uint64_t *)(a + 7 * pb);
    const uint64_t *pb_v = (const uint64_t *)b;
    const uint64_t *nb_v = (const uint64_t *)(b + 7 * pb);

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            const int64_t weight = w[i + j];

            __m256i pos_acc = _mm256_setzero_si256();
            __m256i neg_acc = _mm256_setzero_si256();

            const uint64_t *pai = pa + i * words;
            const uint64_t *nai = na + i * words;
            const uint64_t *pbj = pb_v + j * words;
            const uint64_t *nbj = nb_v + j * words;

            // AVX2 main loop: 256 bits per iteration
            for (int k = 0; k < chunks; k++) {
                __m256i va_p = _mm256_loadu_si256((const __m256i *)(pai + k * 4));
                __m256i va_n = _mm256_loadu_si256((const __m256i *)(nai + k * 4));
                __m256i vb_p = _mm256_loadu_si256((const __m256i *)(pbj + k * 4));
                __m256i vb_n = _mm256_loadu_si256((const __m256i *)(nbj + k * 4));

                // p_m = (pa & pb) | (na & nb) -- agree
                __m256i p_m = _mm256_or_si256(
                    _mm256_and_si256(va_p, vb_p),
                    _mm256_and_si256(va_n, vb_n));

                // n_m = (pa & nb) | (na & pb) -- disagree
                __m256i n_m = _mm256_or_si256(
                    _mm256_and_si256(va_p, vb_n),
                    _mm256_and_si256(va_n, vb_p));

                pos_acc = _avx2_popcnt_accum(pos_acc, p_m);
                neg_acc = _avx2_popcnt_accum(neg_acc, n_m);
            }

            int64_t ij_sum = _avx2_hsum_epi64(pos_acc) - _avx2_hsum_epi64(neg_acc);

            // Scalar tail for non-256-bit-aligned dimensions
            for (int k = chunks * 4; k < words; k++) {
                uint64_t p_m = (pai[k] & pbj[k]) | (nai[k] & nbj[k]);
                uint64_t n_m = (pai[k] & nbj[k]) | (nai[k] & pbj[k]);
                ij_sum += (int64_t)__builtin_popcountll(p_m) - (int64_t)__builtin_popcountll(n_m);
            }

            total += weight * ij_sum;
        }
    }
    return total;
}

int64_t lcvdbt_dot_bpmt7_entropy(const uint8_t *a, const uint8_t *b, int dim) {
    const int words = dim / 64;
    const int pb = dim / 8;
    const int chunks = words / 4;
    int64_t total = 0;

    static const int64_t w[] = {1,3,9,27,81,243,729,2187,6561,19683,59049,177147,531441};

    const uint64_t *pa = (const uint64_t *)a;
    const uint64_t *na = (const uint64_t *)(a + 7 * pb);
    const uint64_t *pb_v = (const uint64_t *)b;
    const uint64_t *nb_v = (const uint64_t *)(b + 7 * pb);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            const int64_t weight = w[i + j];

            __m256i pos_acc = _mm256_setzero_si256();
            __m256i neg_acc = _mm256_setzero_si256();

            const uint64_t *pai = pa + i * words;
            const uint64_t *nai = na + i * words;
            const uint64_t *pbj = pb_v + j * words;
            const uint64_t *nbj = nb_v + j * words;

            for (int k = 0; k < chunks; k++) {
                __m256i va_p = _mm256_loadu_si256((const __m256i *)(pai + k * 4));
                __m256i va_n = _mm256_loadu_si256((const __m256i *)(nai + k * 4));
                __m256i vb_p = _mm256_loadu_si256((const __m256i *)(pbj + k * 4));
                __m256i vb_n = _mm256_loadu_si256((const __m256i *)(nbj + k * 4));

                __m256i p_m = _mm256_or_si256(
                    _mm256_and_si256(va_p, vb_p),
                    _mm256_and_si256(va_n, vb_n));
                __m256i n_m = _mm256_or_si256(
                    _mm256_and_si256(va_p, vb_n),
                    _mm256_and_si256(va_n, vb_p));

                pos_acc = _avx2_popcnt_accum(pos_acc, p_m);
                neg_acc = _avx2_popcnt_accum(neg_acc, n_m);
            }

            int64_t ij_sum = _avx2_hsum_epi64(pos_acc) - _avx2_hsum_epi64(neg_acc);

            for (int k = chunks * 4; k < words; k++) {
                uint64_t p_m = (pa[i*words+k] & pb_v[j*words+k]) | (na[i*words+k] & nb_v[j*words+k]);
                uint64_t n_m = (pa[i*words+k] & nb_v[j*words+k]) | (na[i*words+k] & pb_v[j*words+k]);
                ij_sum += (int64_t)__builtin_popcountll(p_m) - (int64_t)__builtin_popcountll(n_m);
            }

            total += weight * ij_sum;
        }
    }
    return total;
}

int64_t lcvdbt_dot_bpmt7_coarse(const uint8_t *a, const uint8_t *b, int dim) {
    const int words = dim / 64;
    const int pb = dim / 8;
    const int chunks = words / 4;
    int64_t total = 0;

    static const int64_t w[] = {1,3,9,27,81,243,729,2187,6561,19683,59049,177147,531441};

    const uint64_t *pa = (const uint64_t *)a;
    const uint64_t *na = (const uint64_t *)(a + 7 * pb);
    const uint64_t *pb_v = (const uint64_t *)b;
    const uint64_t *nb_v = (const uint64_t *)(b + 7 * pb);

    for (int i = 4; i < 7; i++) {
        for (int j = 4; j < 7; j++) {
            const int64_t weight = w[i + j];

            __m256i pos_acc = _mm256_setzero_si256();
            __m256i neg_acc = _mm256_setzero_si256();

            const uint64_t *pai = pa + i * words;
            const uint64_t *nai = na + i * words;
            const uint64_t *pbj = pb_v + j * words;
            const uint64_t *nbj = nb_v + j * words;

            for (int k = 0; k < chunks; k++) {
                __m256i va_p = _mm256_loadu_si256((const __m256i *)(pai + k * 4));
                __m256i va_n = _mm256_loadu_si256((const __m256i *)(nai + k * 4));
                __m256i vb_p = _mm256_loadu_si256((const __m256i *)(pbj + k * 4));
                __m256i vb_n = _mm256_loadu_si256((const __m256i *)(nbj + k * 4));

                __m256i p_m = _mm256_or_si256(
                    _mm256_and_si256(va_p, vb_p),
                    _mm256_and_si256(va_n, vb_n));
                __m256i n_m = _mm256_or_si256(
                    _mm256_and_si256(va_p, vb_n),
                    _mm256_and_si256(va_n, vb_p));

                pos_acc = _avx2_popcnt_accum(pos_acc, p_m);
                neg_acc = _avx2_popcnt_accum(neg_acc, n_m);
            }

            int64_t ij_sum = _avx2_hsum_epi64(pos_acc) - _avx2_hsum_epi64(neg_acc);

            for (int k = chunks * 4; k < words; k++) {
                uint64_t p_m = (pa[i*words+k] & pb_v[j*words+k]) | (na[i*words+k] & nb_v[j*words+k]);
                uint64_t n_m = (pa[i*words+k] & nb_v[j*words+k]) | (na[i*words+k] & pb_v[j*words+k]);
                ij_sum += (int64_t)__builtin_popcountll(p_m) - (int64_t)__builtin_popcountll(n_m);
            }

            total += weight * ij_sum;
        }
    }
    return total;
}

#else /* !LCVDB_AVX2 — scalar fallbacks for WASM and portable builds */

int64_t lcvdbt_dot_ternary7_avx2(const int16_t *a, const int16_t *b, int loops) {
    int64_t sum = 0;
    for (int i = 0; i < loops * 16; i++) sum += (int64_t)a[i] * b[i];
    return sum;
}

int64_t lcvdbt_dot_ternary7(const int16_t *a, const int16_t *b, int dim) {
    int64_t sum = 0;
    for (int i = 0; i < dim; i++) sum += (int64_t)a[i] * b[i];
    return sum;
}

int64_t lcvdbt_l2sq_ternary7(const int16_t *a, const int16_t *b, int dim) {
    int64_t sum = 0;
    for (int i = 0; i < dim; i++) { int32_t d = (int32_t)a[i] - b[i]; sum += (int64_t)d * d; }
    return sum;
}

int64_t lcvdbt_dot_bpmt7(const uint8_t *a, const uint8_t *b, int dim) {
    return lcvdbt_dot_bpmt7_scalar(a, b, dim);
}

int64_t lcvdbt_dot_bpmt7_entropy(const uint8_t *a, const uint8_t *b, int dim) {
    return lcvdbt_dot_bpmt7_scalar(a, b, dim);
}

int64_t lcvdbt_dot_bpmt7_coarse(const uint8_t *a, const uint8_t *b, int dim) {
    return lcvdbt_dot_bpmt7_scalar(a, b, dim);
}

#endif /* LCVDB_AVX2 */
