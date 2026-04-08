/* platform.h — Platform abstraction for LCVDB
 *
 * Provides compile-time selection of:
 *   - SIMD backend (AVX2 / WASM SIMD / scalar)
 *   - Aligned allocation
 *   - Prefetch hints
 *
 * Compile flags:
 *   -mavx2              → AVX2 backend (native x86)
 *   -msimd128           → WASM SIMD backend (Emscripten)
 *   (neither)           → Scalar fallback
 */

#ifndef LCVDB_PLATFORM_H
#define LCVDB_PLATFORM_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── SIMD backend detection ── */

#if defined(__AVX2__)
    #define LCVDB_AVX2 1
    #include <immintrin.h>
#elif defined(__wasm_simd128__)
    #define LCVDB_WASM_SIMD 1
    #include <wasm_simd128.h>
#else
    #define LCVDB_SCALAR 1
#endif

/* ── Aligned allocation ── */

static inline void *lcvdb_aligned_alloc(size_t alignment, size_t size) {
#if defined(__EMSCRIPTEN__) || defined(LCVDB_SCALAR)
    /* Emscripten: aligned_alloc is C11, supported.
     * Scalar: use aligned_alloc if available, else over-allocate. */
    void *p = NULL;
    size = (size + alignment - 1) & ~(alignment - 1); /* round up to alignment */
    #if defined(_ISOC11_SOURCE) || defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
        p = aligned_alloc(alignment, size);
    #else
        if (posix_memalign(&p, alignment, size) != 0) p = NULL;
    #endif
    return p;
#else
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
#endif
}

/* ── Prefetch ── */

#if defined(__EMSCRIPTEN__) || defined(LCVDB_SCALAR)
    #define lcvdb_prefetch(addr, ...) ((void)0)
#else
    #define lcvdb_prefetch(addr, ...) __builtin_prefetch(addr, ##__VA_ARGS__)
#endif

/* ── Popcount ── */

static inline int lcvdb_popcount64(uint64_t x) {
#if defined(__EMSCRIPTEN__) || defined(LCVDB_SCALAR)
    /* __builtin_popcountll works on Emscripten and GCC/Clang */
    return __builtin_popcountll(x);
#else
    return __builtin_popcountll(x);
#endif
}

/* ── Hamming distance (portable) ── */

static inline int32_t lcvdb_hamming_portable(const uint8_t *a, const uint8_t *b, int bytes) {
#if defined(LCVDB_AVX2)
    const __m256i lo_mask = _mm256_set1_epi8(0x0F);
    const __m256i lut = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    __m256i acc = _mm256_setzero_si256();
    int i;
    for (i = 0; i + 32 <= bytes; i += 32) {
        __m256i x = _mm256_xor_si256(
            _mm256_loadu_si256((const __m256i *)(a + i)),
            _mm256_loadu_si256((const __m256i *)(b + i)));
        __m256i lo = _mm256_and_si256(x, lo_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), lo_mask);
        __m256i pc = _mm256_add_epi8(
            _mm256_shuffle_epi8(lut, lo),
            _mm256_shuffle_epi8(lut, hi));
        acc = _mm256_add_epi64(acc,
            _mm256_sad_epu8(pc, _mm256_setzero_si256()));
    }
    __m128i lo128 = _mm256_castsi256_si128(acc);
    __m128i hi128 = _mm256_extracti128_si256(acc, 1);
    __m128i sum = _mm_add_epi64(lo128, hi128);
    int32_t result = (int32_t)(_mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1));
    /* Scalar tail */
    for (; i < bytes; i++)
        result += __builtin_popcount(a[i] ^ b[i]);
    return result;

#elif defined(LCVDB_WASM_SIMD)
    v128_t acc = wasm_i64x2_const(0, 0);
    const v128_t lo_mask = wasm_i8x16_const_splat(0x0F);
    const v128_t lut = wasm_i8x16_const(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    int i;
    for (i = 0; i + 16 <= bytes; i += 16) {
        v128_t x = wasm_v128_xor(
            wasm_v128_load((const v128_t *)(a + i)),
            wasm_v128_load((const v128_t *)(b + i)));
        v128_t lo = wasm_v128_and(x, lo_mask);
        v128_t hi = wasm_v128_and(wasm_u16x8_shr(x, 4), lo_mask);
        v128_t pc = wasm_i8x16_add(
            wasm_i8x16_swizzle(lut, lo),
            wasm_i8x16_swizzle(lut, hi));
        /* Sum bytes via sad_u8 is not available in WASM SIMD.
         * Use widen + horizontal add instead. */
        v128_t lo16 = wasm_u16x8_extend_low_u8x16(pc);
        v128_t hi16 = wasm_u16x8_extend_high_u8x16(pc);
        v128_t sum16 = wasm_i16x8_add(lo16, hi16);
        /* Horizontal sum of 8 uint16 values */
        v128_t sum32 = wasm_i32x4_add(
            wasm_u32x4_extend_low_u16x8(sum16),
            wasm_u32x4_extend_high_u16x8(sum16));
        acc = wasm_i64x2_add(acc,
            wasm_i64x2_add(
                wasm_u64x2_extend_low_u32x4(sum32),
                wasm_u64x2_extend_high_u32x4(sum32)));
    }
    int32_t result = (int32_t)(wasm_i64x2_extract_lane(acc, 0) +
                                wasm_i64x2_extract_lane(acc, 1));
    for (; i < bytes; i++)
        result += __builtin_popcount(a[i] ^ b[i]);
    return result;

#else
    /* Pure scalar */
    int32_t result = 0;
    int i;
    /* Process 8 bytes at a time via uint64 popcount */
    for (i = 0; i + 8 <= bytes; i += 8) {
        uint64_t xa, xb;
        memcpy(&xa, a + i, 8);
        memcpy(&xb, b + i, 8);
        result += __builtin_popcountll(xa ^ xb);
    }
    for (; i < bytes; i++)
        result += __builtin_popcount(a[i] ^ b[i]);
    return result;
#endif
}

#endif /* LCVDB_PLATFORM_H */
