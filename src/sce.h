/* sce.h — Spatial Compute Engine
 *
 * A fused scan-rerank kernel for nearest-neighbor search in trit Hamming space.
 * No graph. No index. No tree. Just: encode → scan → heap → dot → done.
 *
 * The bounded-rank property of 768-dim trit space (max GT NN rank ≤ 30)
 * guarantees that a fixed shortlist of 50 suffices at any dataset size.
 * The concentration of measure IS the index.
 *
 * Five algebraic primitives fused into one kernel:
 *   1. Addressing: float → MTF7 → trit fingerprint (192 bytes)
 *   2. Distance:   XOR + popcount (Hamming on trits)
 *   3. Concentration: bounded max-heap (shortlist=50)
 *   4. Precision:  dot-product rerank on MTF7 vectors
 *   5. Transport:  optional XOR-delta candidate injection
 */

#ifndef LCVDB_SCE_H
#define LCVDB_SCE_H

#include <stdint.h>
#include "platform.h"
#include "lcvdb_ternary.h"

#define SCE_TRIT_BYTES_768  192     /* 768 dims × 2 bits / 8 */
#define SCE_TRIT_BYTES_256  64      /* 256 dims × 2 bits / 8 */
#define SCE_SHORTLIST       50      /* bounded by concentration of measure */
#define SCE_TRIT_THRESHOLD  3       /* SDF collapse threshold */
#define SCE_MAX_TILES       256

/* ── Filter parameters for pre-filter during scan ── */
typedef struct {
    uint32_t tag;           /* 0 = no tag filter */
    uint32_t bits_lo;       /* 0 = no bitmap filter (lo 32 bits) */
    uint32_t bits_hi;       /* 0 = no bitmap filter (hi 32 bits) */
    int32_t  range_min;     /* only checked if has_range=1 */
    int32_t  range_max;
    int      has_range;     /* 0 = no range filter */
} sce_filter_t;

/* ── Production Parallel SCE API ── */
int sce_search_parallel(
    const float   *query,
    const uint8_t *trit_db,
    const uint8_t *vec_db,
    const float   *norms,
    const lcvdbt_topo_t *topo, /* For filtering */
    uint32_t       n,
    uint16_t       dim,
    int            k,
    int            trit_bytes,
    int            num_threads,
    const sce_filter_t *filter, /* NULL = no filter */
    uint32_t      *result_ids,
    float         *result_scores
);

void sce_encode_database(
    const float   *vectors,
    uint32_t       n,
    uint16_t       dim,
    uint8_t       *trit_out,
    int            trit_bytes,
    uint8_t       *vec_out,
    float         *norms_out
);

/* ── Portable Hamming distance (dispatches via platform.h) ── */

static inline int32_t sce_hamming192(const uint8_t *a, const uint8_t *b) {
    return lcvdb_hamming_portable(a, b, 192);
}

static inline int32_t sce_hamming64(const uint8_t *a, const uint8_t *b) {
    return lcvdb_hamming_portable(a, b, 64);
}

/* ── XOR Transport: apply delta to trit address ── */

static inline void sce_xor_transport(const uint8_t *addr, const uint8_t *delta,
                                      uint8_t *result, int trit_bytes) {
    for (int i = 0; i < trit_bytes; i++)
        result[i] = addr[i] ^ delta[i];
}

/* ── Trit Encode: float nibble → 2-bit trit ── */

static inline void sce_encode_trit(const uint8_t *nibble_fp, int dim,
                                    uint8_t *trit_out) {
    for (int b = 0; b < dim / 4; b++) {
        uint8_t byte = 0;
        for (int t = 0; t < 4; t++) {
            int d = b * 4 + t;
            if (d >= dim) break;
            uint8_t nib = ((d & 1) == 0)
                ? (nibble_fp[d >> 1] >> 4)
                : (nibble_fp[d >> 1] & 0x0F);
            int val = (int)nib - 8;
            uint8_t trit = (val > SCE_TRIT_THRESHOLD) ? 1
                         : (val < -SCE_TRIT_THRESHOLD) ? 2
                         : 0;
            byte |= (trit << (t * 2));
        }
        trit_out[b] = byte;
    }
}

/* ── Max-Heap for bounded shortlist ── */

typedef struct { int32_t ham; uint32_t id; } sce_cand_t;

static inline void sce_heap_sift(sce_cand_t *h, int n, int i) {
    while (1) {
        int l = 2*i+1, r = 2*i+2, lg = i;
        if (l < n && h[l].ham > h[lg].ham) lg = l;
        if (r < n && h[r].ham > h[lg].ham) lg = r;
        if (lg == i) break;
        sce_cand_t t = h[i]; h[i] = h[lg]; h[lg] = t;
        i = lg;
    }
}

static inline void sce_heap_build(sce_cand_t *h, int n) {
    for (int j = n/2 - 1; j >= 0; j--)
        sce_heap_sift(h, n, j);
}

#endif /* LCVDB_SCE_H */
