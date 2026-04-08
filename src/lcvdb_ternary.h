/* ==========================================================================
 * L-Cache VDB — Quantized Vector HNSW (uint32 IDs, runtime-configurable dim)
 * ==========================================================================
 * HNSW algorithm with quantized vectors. Two quantization levels:
 *
 *   Ternary (2 bits/dim): {-1, 0, +1} packed 4 trits per byte, MSB-first.
 *     00 = 0, 01 = +1, 11 = -1 (bit 0 = nonzero, bit 1 = sign).
 *     Distance: AND/XOR/BIC + popcount (dot, cosine, L2).
 *
 *   Binary (1 bit/dim): {0, 1} packed 8 bits per byte, MSB-first.
 *     Bit = 1 if source float > 0, else 0.
 *     Distance: XOR + popcount (Hamming similarity = dim - 2*hamming_dist).
 *
 * Dimension is configurable at init time (default 768D, any multiple of 64).
 * Quantization level is selectable via lcvdbt_init_dim_binary() or
 * lcvdbt_set_quant(). Default is ternary (LCVDBT_QUANT_TERNARY = 2).
 *
 * P18: Node IDs widened from uint16 to uint32. Max nodes = ~4 billion.
 * Topology node expanded from 64B to 128B (2 cache lines).
 * Database header expanded from 64B to 128B (2 cache lines).
 * Generation-counter visited bitmap eliminates VLA stack allocation.
 *
 * Distance metrics: dot product, cosine (pre-normalized), squared L2.
 * Ternary uses AND/XOR + popcount, binary uses XOR + popcount.
 * ========================================================================== */

#ifndef LCVDB_TERNARY_H
#define LCVDB_TERNARY_H

#include <stdint.h>
#include <stddef.h>

/* ---------- Default configuration ----------
 * These compile-time constants are used as defaults when calling lcvdbt_init()
 * without specifying a dimension. They also provide backward compatibility
 * with code that references LCVDBT_VEC_DIM/BYTES directly. */
#ifndef LCVDBT_VEC_DIM
#define LCVDBT_VEC_DIM      768     /* default ternary dimensions per vector */
#endif
#ifndef LCVDBT_VEC_BYTES
#define LCVDBT_VEC_BYTES    192     /* default VEC_DIM * 2 bits / 8          */
#endif

#define LCVDBT_TOPO_SIZE    128     /* bytes per topology node (2 cache lines) */
/* Retained for topo_t on-disk layout — do not modify without migration */
#define LCVDBT_M            16      /* max neighbors per node (layer 0)     */
#define LCVDBT_M_UPPER      4       /* max neighbors per upper layer        */
#define LCVDBT_UPPER_SLOTS  6       /* total upper-layer edge slots         */
#define LCVDBT_MAX_LAYERS   4       /* maximum HNSW layers (0-3)            */
#define LCVDBT_MAX_NODES    0xFFFFFFFE  /* max nodes (uint32 IDs, -1 reserved) */
#define LCVDBT_MAX_TILES    4       /* max emergent routing tiles           */
#define LCVDBT_ROUTE_TOP    3       /* search top-N tiles during query      */

/* ---------- Distance metric ---------- */
#define LCVDBT_METRIC_DOT     0   /* dot product: higher = closer (default)  */
#define LCVDBT_METRIC_COSINE  1   /* cosine: dot on pre-normalized vectors   */
#define LCVDBT_METRIC_L2      2   /* squared L2: lower = closer (negated internally) */

/* ---------- Quantization level ---------- */
#define LCVDBT_QUANT_TERNARY  2   /* 2 bits/dim: {-1, 0, +1} interleaved    */
#define LCVDBT_QUANT_BINARY   1   /* 1 bit/dim:  {0, 1} sign-only           */
#define LCVDBT_QUANT_SPLIT    3   /* 2 bits/dim: {-1, 0, +1} split-plane    */
#define LCVDBT_QUANT_TERNARY4 4   /* 81 levels, ~6.3 bits, stored as int8    */
#define LCVDBT_QUANT_TERNARY7 5   /* 2187 levels, ~11 bits, stored as int16 (deprecated) */
#define LCVDBT_QUANT_BPMT7    6   /* 2187 levels, bit-parallel true ternary */
#define LCVDBT_QUANT_MTF7     7   /* 2187 levels, bit-parallel ternary + per-vector scale */

/* ---------- Node flags ---------- */
#define LCVDBT_FLAG_DELETED  0x0001

/* ---------- Invalid ID sentinel ---------- */
#define LCVDBT_INVALID_ID    0xFFFFFFFFu

/* ---------- Dimension helpers ---------- */
#define LCVDBT_DIM_TO_VEC_BYTES(dim)   ((uint16_t)((dim) / 4))
#define LCVDBT_DIM_TO_SIGN_BYTES(dim)  ((uint16_t)((dim) / 8))
#define LCVDBT_DIM_TO_VEC_LOOPS(dim)   ((int)((dim) / 64))
#define LCVDBT_DIM_TO_SIGN_LOOPS(dim)  ((int)((dim) / 128))
#define LCVDBT_DIM_TO_F32_LOOPS(dim)   ((int)((dim) / 16))

#define LCVDBT_QUANT_VEC_BYTES(dim, quant)  \
    ((uint16_t)((quant) == LCVDBT_QUANT_TERNARY4 ? (dim) : \
                (quant) == LCVDBT_QUANT_TERNARY7 ? ((dim) * 2) : \
                (quant) == LCVDBT_QUANT_BPMT7    ? ((dim) * 14 / 8) : \
                (quant) == LCVDBT_QUANT_MTF7     ? ((dim) * 14 / 8) : \
                ((dim) * ((quant) == LCVDBT_QUANT_SPLIT ? 2 : (quant)) / 8)))
#define LCVDBT_QUANT_VEC_LOOPS(dim, quant)  \
    ((int)((quant) == LCVDBT_QUANT_TERNARY4 ? ((dim) / 16) : \
           (quant) == LCVDBT_QUANT_TERNARY7 ? ((dim) / 8) : \
           (quant) == LCVDBT_QUANT_BPMT7    ? ((dim) / 64) : \
           (quant) == LCVDBT_QUANT_MTF7     ? ((dim) / 64) : \
           ((dim) * ((quant) == LCVDBT_QUANT_SPLIT ? 2 : (quant)) / 8 / 16)))

#define LCVDBT_SPLIT_PLANE_BYTES(dim)  ((uint16_t)((dim) / 8))
#define LCVDBT_SPLIT_PLANE_LOOPS(dim)  ((int)((dim) / 128))

/* ---------- Topology Node ---------- */
typedef struct __attribute__((aligned(64))) {
    uint32_t neighbors[LCVDBT_M];                /* [0..63]   layer-0 edges     */
    uint8_t  neighbor_count;                      /* [64]      layer-0 count     */
    uint8_t  max_layer;                           /* [65]                        */
    uint16_t flags;                               /* [66..67]                    */
    uint32_t payload_id;                          /* [68..71]                    */
    uint32_t upper_neighbors[LCVDBT_UPPER_SLOTS]; /* [72..95]  upper-layer edges */
    uint8_t  upper_count[LCVDBT_MAX_LAYERS - 1];  /* [96..98]  per-layer counts  */
    uint8_t  tile_id;                             /* [99]      assigned tile     */
    uint32_t filter_bits_lo;                      /* [100..103] categories 0-31   */
    uint32_t filter_bits_hi;                      /* [104..107] categories 32-63  */
    uint32_t filter_tag;                          /* [108..111] equality tag      */
    int32_t  filter_min;                          /* [112..115] range minimum     */
    int32_t  filter_max;                          /* [116..119] range maximum     */
    uint8_t  _reserved[8];                        /* [120..127] pad to 128 bytes */
} lcvdbt_topo_t;

_Static_assert(sizeof(lcvdbt_topo_t) == 128, "Topo node must be 2 cache lines (128 bytes)");

/* ---------- Database ---------- */
typedef struct __attribute__((aligned(64))) {
    uint32_t node_count;                        /* [0..3]     */
    uint32_t entry_point;                       /* [4..7]     */
    uint8_t  max_level;                         /* [8]        */
    uint8_t  M;                                 /* [9]        */
    uint16_t vec_dim;                           /* [10..11]   runtime dimension  */
    uint32_t max_nodes;                         /* [12..15]   */
    lcvdbt_topo_t *topo_array;                  /* [16..23]   */
    uint8_t       *vec_array;                   /* [24..31]   raw packed vectors */
    uint32_t prng_state;                        /* [32..35]   */
    uint32_t free_head;                         /* [36..39]   */
    uint32_t free_count;                        /* [40..43]   */
    uint8_t  num_tiles;                         /* [44]       */
    uint8_t  metric;                            /* [45]       distance metric    */
    uint8_t  quant;                             /* [46]       quantization level */
    uint8_t  _managed;                          /* [47]       1 if managed alloc  */
    uint32_t tile_count[LCVDBT_MAX_TILES];      /* [48..63]   nodes per tile     */
    uint32_t tile_entry[LCVDBT_MAX_TILES];      /* [64..79]   entry point / tile */
    float    *mtf_scales;                       /* [80..87]   Per-node MTF7 pack scales */
    float    *quant_norms;                      /* [88..95]   Per-node normalization factors */
    void    *sign_array;                        /* [96..103]  optional sign reps */
    uint8_t *visited_buf;                       /* [104..111] gen-counter bitmap */
    void    *hooks;                             /* [112..119] runtime hooks      */
    uint32_t visited_gen;                       /* [120..123] current generation */
    uint8_t  mtf_construction_raw;              /* [124]       use raw dot for MTF build */
    uint8_t  mtf_staged_proxy;                  /* [125]       staged proxy construction mode */
    uint8_t  save_fmt;                           /* [126]      0=legacy, 1=+scales */
    uint8_t  _reserved[1];                      /* [127]      pad to 128 bytes   */
} lcvdbt_t;

_Static_assert(sizeof(lcvdbt_t) == 128, "DB struct must be exactly 2 cache lines (128 bytes)");

/* ---------- Concurrency Hooks ---------- */
typedef void (*lcvdbt_lock_fn)(void *ctx);
typedef struct {
    void *ctx;
    lcvdbt_lock_fn lock_read;
    lcvdbt_lock_fn lock_write;
    lcvdbt_lock_fn unlock;
} lcvdbt_hooks_t;

void lcvdbt_set_hooks(lcvdbt_t *db, const lcvdbt_hooks_t *hooks);

#define LCVDBT_LOCK_READ(db) do { \
    lcvdbt_hooks_t *h = (lcvdbt_hooks_t *)(db)->hooks; \
    if (h && h->lock_read) h->lock_read(h->ctx); \
} while(0)

#define LCVDBT_LOCK_WRITE(db) do { \
    lcvdbt_hooks_t *h = (lcvdbt_hooks_t *)(db)->hooks; \
    if (h && h->lock_write) h->lock_write(h->ctx); \
} while(0)

#define LCVDBT_UNLOCK(db) do { \
    lcvdbt_hooks_t *h = (lcvdbt_hooks_t *)(db)->hooks; \
    if (h && h->unlock) h->unlock(h->ctx); \
} while(0)

/* ---------- Thread-Local Search Context ---------- */
typedef struct {
    uint8_t *visited_buf;
    uint32_t  visited_gen;
} lcvdbt_search_ctx_t;

static inline void lcvdbt_search_ctx_init(lcvdbt_search_ctx_t *ctx, uint8_t *visited_buf) {
    ctx->visited_buf = visited_buf;
    ctx->visited_gen = 1;
}

static inline void lcvdbt_search_ctx_clear(lcvdbt_search_ctx_t *ctx, uint32_t max_nodes) {
    ctx->visited_gen++;
    if ((ctx->visited_gen & 0xFF) == 0) {
        __builtin_memset(ctx->visited_buf, 0, max_nodes);
        ctx->visited_gen = 1;
    }
}

/* ---------- Search API (HNSW — used by build_ternary.c insert path) ---------- */
/* ---------- Accessor macros ---------- */
#define lcvdbt_vec_bytes(db)      LCVDBT_QUANT_VEC_BYTES((db)->vec_dim, (db)->quant)
#define lcvdbt_sign_bytes(db)     LCVDBT_DIM_TO_SIGN_BYTES((db)->vec_dim)
#define lcvdbt_vec_loops(db)      LCVDBT_QUANT_VEC_LOOPS((db)->vec_dim, (db)->quant)
#define lcvdbt_sign_loops(db)     LCVDBT_DIM_TO_SIGN_LOOPS((db)->vec_dim)
#define lcvdbt_f32_loops(db)      LCVDBT_DIM_TO_F32_LOOPS((db)->vec_dim)
#define lcvdbt_vec_at(db, id)     ((db)->vec_array + (size_t)(id) * lcvdbt_vec_bytes(db))

#define lcvdbt_sign_stride(db)    (((lcvdbt_sign_bytes(db) + 63) & ~63))
#define lcvdbt_sign_at(db, id)    ((uint8_t*)(db)->sign_array + (size_t)(id) * lcvdbt_sign_stride(db))
#define lcvdbt_mtf_fp_stride(db)  ((((db)->vec_dim) + 1) / 2)
#define lcvdbt_proxy_stride(db)   LCVDBT_QUANT_VEC_BYTES((db)->vec_dim, LCVDBT_QUANT_BPMT7)
#define lcvdbt_mtf_aux_stride(db) (lcvdbt_proxy_stride(db) + lcvdbt_mtf_fp_stride(db))
#define lcvdbt_proxy_at(db, id)   ((uint8_t*)(db)->sign_array + (size_t)(id) * lcvdbt_mtf_aux_stride(db))
#define lcvdbt_mtf_fp_at(db, id)  (lcvdbt_proxy_at(db, id) + lcvdbt_proxy_stride(db))

/* ---------- Visited bitmap ---------- */
#define lcvdbt_vis_set(db, id)    ((db)->visited_buf[(id)] = (uint8_t)(db)->visited_gen)
#define lcvdbt_vis_test(db, id)   ((db)->visited_buf[(id)] == (uint8_t)(db)->visited_gen)
#define lcvdbt_vis_clear(db)      do { \
    (db)->visited_gen++; \
    if (((db)->visited_gen & 0xFF) == 0) { \
        __builtin_memset((db)->visited_buf, 0, (db)->max_nodes); \
        (db)->visited_gen = 1; \
    } \
} while(0)

/* ---------- API ---------- */
void lcvdbt_init(lcvdbt_t *db, void *topo_buf, void *vec_buf, void *vis_buf, uint32_t max_nodes);
size_t lcvdbt_sizeof_topo(uint32_t max_nodes);
size_t lcvdbt_sizeof_vec(uint32_t max_nodes, uint16_t dim, uint8_t quant);
size_t lcvdbt_sizeof_vis(uint32_t max_nodes);
lcvdbt_t *lcvdbt_new(uint32_t max_nodes, uint16_t dim, uint8_t quant);
void lcvdbt_free(lcvdbt_t *db);
void lcvdbt_init_dim(lcvdbt_t *db, void *topo_buf, void *vec_buf, void *vis_buf, uint32_t max_nodes, uint16_t dim);
void lcvdbt_set_metric(lcvdbt_t *db, uint8_t metric);
void lcvdbt_set_quant(lcvdbt_t *db, uint8_t quant);

/* Flat insert: O(1) pack + append (no graph construction) */
uint32_t lcvdbt_insert_flat(lcvdbt_t *db, const uint8_t *packed_vec, uint32_t payload_id, float mtf_scale);
uint32_t lcvdbt_insert(lcvdbt_t *db, const uint8_t *packed_vec, uint32_t payload_id);
uint32_t lcvdbt_insert_mtf7(lcvdbt_t *db, const uint8_t *packed_vec, uint32_t payload_id, float mtf_scale);
void lcvdbt_delete(lcvdbt_t *db, uint32_t node_id);
void lcvdbt_update(lcvdbt_t *db, uint32_t node_id, const uint8_t *packed_vec);

static inline void lcvdbt_set_filter(lcvdbt_t *db, uint32_t node_id, uint32_t bits_lo, uint32_t bits_hi, uint32_t tag, int32_t min_val, int32_t max_val) {
    db->topo_array[node_id].filter_bits_lo = bits_lo;
    db->topo_array[node_id].filter_bits_hi = bits_hi;
    db->topo_array[node_id].filter_tag = tag;
    db->topo_array[node_id].filter_min = min_val;
    db->topo_array[node_id].filter_max = max_val;
}

static inline void lcvdbt_set_filter_bits(lcvdbt_t *db, uint32_t node_id, uint32_t bits_lo, uint32_t bits_hi) {
    db->topo_array[node_id].filter_bits_lo = bits_lo;
    db->topo_array[node_id].filter_bits_hi = bits_hi;
}


/* ---------- Distance functions ---------- */
int32_t lcvdbt_dot_ternary(const uint8_t *a, const uint8_t *b, int loops);
int32_t lcvdbt_dot_ternary_ext(const uint8_t *a, const uint8_t *b, int32_t *support_out, int loops);
int32_t lcvdbt_l2sq_ternary(const uint8_t *a, const uint8_t *b, int loops);
int64_t lcvdbt_dot_ternary4(const int8_t *a, const int8_t *b, int dim);
int64_t lcvdbt_dot_ternary4_avx2(const int8_t *a, const int8_t *b, int loops);
int32_t lcvdbt_l2sq_ternary4(const int8_t *a, const int8_t *b, int dim);
int64_t lcvdbt_dot_ternary7(const int16_t *a, const int16_t *b, int dim);
int64_t lcvdbt_dot_ternary7_avx2(const int16_t *a, const int16_t *b, int loops);
int64_t lcvdbt_l2sq_ternary7(const int16_t *a, const int16_t *b, int dim);
int64_t lcvdbt_dot_bpmt7(const uint8_t *a, const uint8_t *b, int dim);
int64_t lcvdbt_dot_bpmt7_scalar(const uint8_t *a, const uint8_t *b, int dim);
int64_t lcvdbt_dot_bpmt7_coarse(const uint8_t *a, const uint8_t *b, int dim);
int64_t lcvdbt_dot_bpmt7_entropy(const uint8_t *a, const uint8_t *b, int dim);
void lcvdbt_pack_f32_bpmt7(const float *src, uint8_t *dst, int dim);
void lcvdbt_pack_f32_mtf7(const float *src, uint8_t *dst, int dim, float *out_scale);
void lcvdbt_repack_mtf7_proxy(const uint8_t *src, float scale, uint8_t *dst, int dim);
void lcvdbt_pack_mtf7_fingerprint(const uint8_t *src, uint8_t *dst, int dim);
int32_t lcvdbt_dot_split(const uint8_t *a, const uint8_t *b, int plane_loops);
int32_t lcvdbt_dot_split_ext(const uint8_t *a, const uint8_t *b, int32_t *support_out, int plane_loops);
int32_t lcvdbt_l2sq_split(const uint8_t *a, const uint8_t *b, int plane_loops);

/* ---------- Metric-aware scoring ---------- */
#define COMPOSITE_SCORE(dot, support) ((int32_t)((dot) * 1024 + (support)))
#define COMPOSITE_DOT(cs) ((cs) / 1024)

static inline int32_t metric_score_ext_q(uint8_t metric, uint8_t quant,
                                          const uint8_t *a, const uint8_t *b,
                                          int32_t *support, int loops,
                                          uint16_t dim, int coarse) {
    if (quant == LCVDBT_QUANT_SPLIT) {
        int pl = loops >> 1;
        if (metric == LCVDBT_METRIC_L2) {
            *support = 0;
            return -lcvdbt_l2sq_split(a, b, pl);
        }
        int32_t d = lcvdbt_dot_split_ext(a, b, support, pl);
        return COMPOSITE_SCORE(d, *support);
    }
    if (quant == LCVDBT_QUANT_TERNARY4) {
        *support = dim;
        if (metric == LCVDBT_METRIC_L2)
            return -lcvdbt_l2sq_ternary4((const int8_t *)a, (const int8_t *)b, (int)dim);
        return (int32_t)lcvdbt_dot_ternary4((const int8_t *)a, (const int8_t *)b, (int)dim);
    }
    if (quant == LCVDBT_QUANT_TERNARY7) {
        int64_t dot = lcvdbt_dot_ternary7((const int16_t *)a, (const int16_t *)b, (int)dim);
        *support = dim;
        if (metric == LCVDBT_METRIC_L2) {
            int64_t l2 = lcvdbt_l2sq_ternary7((const int16_t *)a, (const int16_t *)b, (int)dim);
            return -(int32_t)(l2 >> 16);
        }
        return (int32_t)(dot >> 8);
    }
    if (quant == LCVDBT_QUANT_BPMT7) {
        // Atomic Fix: Disable coarse navigation.
        // Bit-parallel logic is fast enough (24ns) to use full 7-trit precision
        // for every hop. This ensures perfect navigation (96%+ R@1).
        int64_t dot = lcvdbt_dot_bpmt7(a, b, (int)dim);
        *support = dim;
        return (int32_t)dot;
    }
    if (quant == LCVDBT_QUANT_MTF7) {
        int64_t dot = lcvdbt_dot_bpmt7(a, b, (int)dim);
        *support = dim;
        return (int32_t)dot;
    }
    if (metric == LCVDBT_METRIC_L2) {
        *support = 0;
        return -lcvdbt_l2sq_ternary(a, b, loops);
    }
    int32_t d = lcvdbt_dot_ternary_ext(a, b, support, loops);
    return COMPOSITE_SCORE(d, *support);
}

static inline int32_t metric_score_q(uint8_t metric, uint8_t quant,
                                      const uint8_t *a, const uint8_t *b,
                                      int loops, uint16_t dim, int coarse) {
    if (quant == LCVDBT_QUANT_SPLIT) {
        int pl = loops >> 1;
        if (metric == LCVDBT_METRIC_L2)
            return -lcvdbt_l2sq_split(a, b, pl);
        return lcvdbt_dot_split(a, b, pl);
    }
    if (quant == LCVDBT_QUANT_TERNARY4) {
        if (metric == LCVDBT_METRIC_L2)
            return -lcvdbt_l2sq_ternary4((const int8_t *)a, (const int8_t *)b, (int)dim);
        return (int32_t)lcvdbt_dot_ternary4((const int8_t *)a, (const int8_t *)b, (int)dim);
    }
    if (quant == LCVDBT_QUANT_TERNARY7) {
        int64_t dot = lcvdbt_dot_ternary7((const int16_t *)a, (const int16_t *)b, (int)dim);
        if (metric == LCVDBT_METRIC_L2) {
            int64_t l2 = lcvdbt_l2sq_ternary7((const int16_t *)a, (const int16_t *)b, (int)dim);
            return -(int32_t)(l2 >> 16);
        }
        return (int32_t)(dot >> 8);
    }
    if (quant == LCVDBT_QUANT_BPMT7) {
        return (int32_t)(coarse ? lcvdbt_dot_bpmt7_coarse(a, b, (int)dim) :
                                   lcvdbt_dot_bpmt7(a, b, (int)dim));
    }
    if (quant == LCVDBT_QUANT_MTF7) {
        return (int32_t)(coarse ? lcvdbt_dot_bpmt7_coarse(a, b, (int)dim) :
                                   lcvdbt_dot_bpmt7(a, b, (int)dim));
    }
    if (metric == LCVDBT_METRIC_L2)
        return -lcvdbt_l2sq_ternary(a, b, loops);
    return lcvdbt_dot_ternary(a, b, loops);
}

/* ---------- Persistence ---------- */
int lcvdbt_save(const lcvdbt_t *db, int fd);
int lcvdbt_load(lcvdbt_t *db, int fd, void *topo_buf, void *vec_buf, void *vis_buf);
void *lcvdbt_mmap_load(int fd, lcvdbt_t **db_out, void *vis_buf, size_t *file_size);
int lcvdbt_mmap_unload(void *mmap_base, size_t file_size);
static inline size_t lcvdbt_file_size(uint32_t max_nodes, uint16_t dim, uint8_t quant) {
    uint16_t vb = LCVDBT_QUANT_VEC_BYTES(dim, quant);
    return sizeof(lcvdbt_t) + (size_t)max_nodes * sizeof(lcvdbt_topo_t) +
           (size_t)max_nodes * vb + (size_t)max_nodes * sizeof(float); /* +mtf_scales */
}

/* ---------- Packing ---------- */
void lcvdbt_pack_f32_bpmt7(const float *src, uint8_t *dst, int dim);
void lcvdbt_pack_f32_mtf7(const float *src, uint8_t *dst, int dim, float *out_scale);
void lcvdbt_unpack_mtf7_f32(const uint8_t *packed, float scale, float *dst, int dim);
void lcvdbt_repack_mtf7_proxy(const uint8_t *src, float scale, uint8_t *dst, int dim);
void lcvdbt_pack_mtf7_fingerprint(const uint8_t *src, uint8_t *dst, int dim);

/* ---------- MTF21: 3-tier residual cascade (float32-equivalent precision, float-free output) ---------- */
void lcvdbt_pack_f32_mtf21(const float *src, uint8_t *dst, int dim, uint32_t *scales_out);
double lcvdbt_dot_mtf21(const uint8_t *a, const uint32_t *a_scales,
                         const uint8_t *b, const uint32_t *b_scales, int dim);

/* ---------- Sign extraction ---------- */
void lcvdbt_extract_sign(const uint8_t *packed, uint8_t *sign_out, int dim, uint8_t quant);

/* ---------- Upper-layer edge access helpers ---------- */
static inline const uint32_t *lcvdbt_layer_edges(const lcvdbt_topo_t *t, int layer, int *count) {
    if (layer == 0) {
        *count = t->neighbor_count;
        return t->neighbors;
    }
    int offset = 0;
    for (int l = 1; l < layer; l++)
        offset += t->upper_count[l - 1];
    *count = t->upper_count[layer - 1];
    return &t->upper_neighbors[offset];
}

static inline int lcvdbt_upper_offset(const lcvdbt_topo_t *t, int layer) {
    int offset = 0;
    for (int l = 1; l < layer; l++)
        offset += t->upper_count[l - 1];
    return offset;
}

static inline int lcvdbt_upper_remaining(const lcvdbt_topo_t *t) {
    int used = 0;
    for (int l = 0; l < LCVDBT_MAX_LAYERS - 1; l++)
        used += t->upper_count[l];
    return LCVDBT_UPPER_SLOTS - used;
}

#endif /* LCVDB_TERNARY_H */
