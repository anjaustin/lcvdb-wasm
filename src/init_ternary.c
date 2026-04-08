/* L-Cache VDB — Ternary variant initialization (runtime-configurable dim, uint32 IDs) */
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "platform.h"
#include "lcvdb_ternary.h"

/* Internal init shared by ternary and binary variants. */
static void init_common(lcvdbt_t *db, void *topo_buf, void *vec_buf,
                         void *vis_buf, uint32_t max_nodes, uint16_t dim,
                         uint8_t quant) {
    memset(db, 0, sizeof(lcvdbt_t));
    db->M = LCVDBT_M;
    db->entry_point = LCVDBT_INVALID_ID;
    db->topo_array = (lcvdbt_topo_t *)topo_buf;
    db->vec_array = (uint8_t *)vec_buf;
    db->visited_buf = (uint8_t *)vis_buf;
    db->mtf_scales = calloc(max_nodes, sizeof(float));
    db->quant_norms = calloc(max_nodes, sizeof(float));
    db->visited_gen = 1;
    db->max_nodes = max_nodes;
    db->prng_state = 0x12345678;
    db->free_head = LCVDBT_INVALID_ID;
    db->free_count = 0;
    db->vec_dim = dim;
    db->quant = quant;

    uint16_t vec_bytes = LCVDBT_QUANT_VEC_BYTES(dim, quant);
    memset(topo_buf, 0, (size_t)max_nodes * sizeof(lcvdbt_topo_t));
    memset(vec_buf, 0, (size_t)max_nodes * vec_bytes);
    memset(vis_buf, 0, max_nodes);
    if (quant == LCVDBT_QUANT_MTF7) {
        size_t aux_bytes = (size_t)max_nodes * (LCVDBT_QUANT_VEC_BYTES(dim, LCVDBT_QUANT_BPMT7) + ((dim + 1) / 2));
        db->sign_array = calloc(1, aux_bytes);
    }
}

/* Core init with explicit dimension (ternary, 2 bits/dim). */
void lcvdbt_init_dim(lcvdbt_t *db, void *topo_buf, void *vec_buf,
                     void *vis_buf, uint32_t max_nodes, uint16_t dim) {
    init_common(db, topo_buf, vec_buf, vis_buf, max_nodes, dim,
                LCVDBT_QUANT_TERNARY);
}

/* Backward-compatible init: uses default LCVDBT_VEC_DIM. */
void lcvdbt_init(lcvdbt_t *db, void *topo_buf, void *vec_buf,
                 void *vis_buf, uint32_t max_nodes) {
    lcvdbt_init_dim(db, topo_buf, vec_buf, vis_buf, max_nodes, LCVDBT_VEC_DIM);
}

/* Set distance metric. */
void lcvdbt_set_metric(lcvdbt_t *db, uint8_t metric) {
    db->metric = metric;
}

/* Set quantization level. Must be called before any inserts. */
void lcvdbt_set_quant(lcvdbt_t *db, uint8_t quant) {
    db->quant = quant;
}

/* Set MTF7 to use raw (unnormalized) dot product during construction. */
void lcvdbt_set_mtf_construction_raw(lcvdbt_t *db, int use_raw) {
    db->mtf_construction_raw = use_raw ? 1 : 0;
}

void lcvdbt_set_mtf_staged_proxy(lcvdbt_t *db, int use_staged) {
    db->mtf_staged_proxy = use_staged ? 1 : 0;
}

/* Set concurrency hooks. Pass NULL to disable locking. */
void lcvdbt_set_hooks(lcvdbt_t *db, const lcvdbt_hooks_t *hooks) {
    db->hooks = (void *)hooks;
}

/* Initialize sign-only array for upper-layer navigation. */
void lcvdbt_init_signs(lcvdbt_t *db, void *sign_buf) {
    db->sign_array = sign_buf;
    if (sign_buf)
        memset(sign_buf, 0, (size_t)db->max_nodes * lcvdbt_sign_stride(db));
}

/* Initialize tile routing. */

/* Assign vectors to buckets based on proxy hash. */
/* Recompute tile signatures from current vectors. */
/* Pack float32 vector to ternary. */
/* Pack float32 vector to binary (sign-only). */
/* Pack float32 vector to Ternary-4 (81-level balanced ternary). */
/* Initialize with split-plane ternary. */
/* Pack float32 vector to Bit-Parallel Multi-Trit (BPMT7). */
void lcvdbt_pack_f32_bpmt7(const float *src, uint8_t *dst, int dim) {
    const int T7_MAX = 1093;
    int plane_bytes = dim / 8;
    memset(dst, 0, (size_t)plane_bytes * 14);
    for (int d = 0; d < dim; d++) {
        int v = (int)roundf(src[d] * 4000.0f);
        if (v > T7_MAX) v = T7_MAX;
        if (v < -T7_MAX) v = -T7_MAX;
        int byte_idx = d / 8;
        int bit_pos = 7 - (d % 8);
        for (int i = 0; i < 7; i++) {
            int r = v % 3;
            if (r < 0) r += 3;
            if (r == 1) {
                dst[i * plane_bytes + byte_idx] |= (1 << bit_pos);
                v -= 1;
            } else if (r == 2) {
                dst[(i + 7) * plane_bytes + byte_idx] |= (1 << bit_pos);
                v += 1;
            }
            v /= 3;
        }
    }
}

/* Pack float32 vector to Bit-Parallel Multi-Trit with per-vector scale.
 * out_scale stores the mantissa scale factor used for encoding.
 * Original value is approximated by packed_value / out_scale. */
void lcvdbt_pack_f32_mtf7(const float *src, uint8_t *dst, int dim, float *out_scale) {
    const int T7_MAX = 1093;
    float mx = 0.0f;
    int plane_bytes = dim / 8;
    memset(dst, 0, (size_t)plane_bytes * 14);

    for (int i = 0; i < dim; i++) {
        float av = fabsf(src[i]);
        if (av > mx) mx = av;
    }

    float scale = (mx > 1e-12f) ? ((float)T7_MAX / mx) : 1.0f;
    if (out_scale) *out_scale = scale;

    for (int d = 0; d < dim; d++) {
        int v = (int)lrintf(src[d] * scale);
        if (v > T7_MAX) v = T7_MAX;
        if (v < -T7_MAX) v = -T7_MAX;

        int byte_idx = d / 8;
        int bit_pos = 7 - (d % 8);
        for (int i = 0; i < 7; i++) {
            int r = v % 3;
            if (r < 0) r += 3;
            if (r == 1) {
                dst[i * plane_bytes + byte_idx] |= (1 << bit_pos);
                v -= 1;
            } else if (r == 2) {
                dst[(i + 7) * plane_bytes + byte_idx] |= (1 << bit_pos);
                v += 1;
            }
            v /= 3;
        }
    }
}

/* Decode MTF7 packed vector back to float32 (inverse of pack_f32_mtf7). */
void lcvdbt_unpack_mtf7_f32(const uint8_t *packed, float scale, float *dst, int dim) {
    int plane_bytes = dim / 8;
    float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 1.0f;
    static const int pow3[7] = {1, 3, 9, 27, 81, 243, 729};

    for (int d = 0; d < dim; d++) {
        int byte_idx = d / 8;
        int bit_pos = 7 - (d % 8);
        int val = 0;
        for (int i = 0; i < 7; i++) {
            if (packed[i * plane_bytes + byte_idx] & (1 << bit_pos))
                val += pow3[i];       /* positive trit: +3^i */
            else if (packed[(i + 7) * plane_bytes + byte_idx] & (1 << bit_pos))
                val -= pow3[i];       /* negative trit: -3^i */
        }
        dst[d] = (float)val * inv_scale;
    }
}

/* Encode float32 vector as MTF21 (3-tier residual cascade, float-free output).
 *
 * Output:
 *   dst = [tier0: vb][tier1: vb][tier2: vb]  (pure ternary, 4,032 bytes at 768D)
 *   scales_out = [s0, s1, s2]  (uint32 mantissas, no float)
 *
 * Scale reconstruction:
 *   tier 0 real_scale = scales_out[0]
 *   tier 1 real_scale = scales_out[1] × 3^7   (= × 2187)
 *   tier 2 real_scale = scales_out[2] × 3^14  (= × 4782969)
 *
 * Zero float32 in the output. Float is used only during encoding computation.
 */
#define MTF21_TIER_BASE  2187    /* 3^7 */
#define MTF21_TIER2_BASE 4782969 /* 3^14 */

void lcvdbt_pack_f32_mtf21(const float *src, uint8_t *dst, int dim, uint32_t *scales_out) {
    int vb = (dim * 14) / 8;
    float *residual = (float *)malloc((size_t)dim * sizeof(float));
    float *decoded = (float *)malloc((size_t)dim * sizeof(float));
    float fscale;

    /* Tier 0 */
    lcvdbt_pack_f32_mtf7(src, dst, dim, &fscale);
    scales_out[0] = (uint32_t)(fscale + 0.5f);
    lcvdbt_unpack_mtf7_f32(dst, fscale, decoded, dim);
    for (int i = 0; i < dim; i++) residual[i] = src[i] - decoded[i];

    /* Tier 1 */
    lcvdbt_pack_f32_mtf7(residual, dst + vb, dim, &fscale);
    scales_out[1] = (uint32_t)(fscale / MTF21_TIER_BASE + 0.5f);
    lcvdbt_unpack_mtf7_f32(dst + vb, fscale, decoded, dim);
    for (int i = 0; i < dim; i++) residual[i] -= decoded[i];

    /* Tier 2 */
    lcvdbt_pack_f32_mtf7(residual, dst + 2 * vb, dim, &fscale);
    scales_out[2] = (uint32_t)(fscale / MTF21_TIER2_BASE + 0.5f);

    free(residual);
    free(decoded);
}

/* Full-precision MTF21 dot product. Scales are uint32 mantissas (no float stored).
 * Decodes per-dimension to avoid catastrophic cancellation. */
double lcvdbt_dot_mtf21(const uint8_t *a, const uint32_t *a_scales,
                         const uint8_t *b, const uint32_t *b_scales, int dim) {
    int vb = (dim * 14) / 8;
    int plane_bytes = dim / 8;
    static const int pow3[7] = {1, 3, 9, 27, 81, 243, 729};
    static const double tier_base[3] = {1.0, (double)MTF21_TIER_BASE, (double)MTF21_TIER2_BASE};
    double dot = 0.0;

    for (int d = 0; d < dim; d++) {
        int byte_idx = d / 8;
        int bit_pos = 7 - (d % 8);
        double av = 0.0, bv = 0.0;

        for (int tier = 0; tier < 3; tier++) {
            const uint8_t *ap = a + tier * vb;
            const uint8_t *bp = b + tier * vb;
            double a_real_scale = (double)a_scales[tier] * tier_base[tier];
            double b_real_scale = (double)b_scales[tier] * tier_base[tier];
            int aval = 0, bval = 0;
            for (int i = 0; i < 7; i++) {
                if (ap[i * plane_bytes + byte_idx] & (1 << bit_pos))
                    aval += pow3[i];
                else if (ap[(i + 7) * plane_bytes + byte_idx] & (1 << bit_pos))
                    aval -= pow3[i];
                if (bp[i * plane_bytes + byte_idx] & (1 << bit_pos))
                    bval += pow3[i];
                else if (bp[(i + 7) * plane_bytes + byte_idx] & (1 << bit_pos))
                    bval -= pow3[i];
            }
            av += (double)aval / a_real_scale;
            bv += (double)bval / b_real_scale;
        }
        dot += av * bv;
    }
    return dot;
}

/* Extract sign-only representation from packed ternary. */
void lcvdbt_extract_sign(const uint8_t *packed, uint8_t *sign_out, int dim, uint8_t quant) {
    int sign_bytes = dim / 8;
    memset(sign_out, 0, (size_t)sign_bytes);
    if (quant == LCVDBT_QUANT_BPMT7 || quant == LCVDBT_QUANT_MTF7) {
        int plane_bytes = dim / 8;
        for (int d = 0; d < dim; d++) {
            int byte_idx = d / 8;
            int bit_pos = 7 - (d % 8);
            int sign = 0;
            for (int i = 6; i >= 0; i--) {
                if (packed[i * plane_bytes + byte_idx] & (1 << bit_pos)) {
                    sign = 1; break;
                }
                if (packed[(i + 7) * plane_bytes + byte_idx] & (1 << bit_pos)) {
                    sign = -1; break;
                }
            }
            if (sign == 1) sign_out[byte_idx] |= (1 << bit_pos);
        }
        return;
    }
    for (int i = 0; i < dim; i++) {
        int byte_idx = i / 4;
        int shift = (3 - (i % 4)) * 2;
        uint8_t trit = (packed[byte_idx] >> shift) & 0x03;
        if (trit == 0x01) {
             sign_out[i/8] |= (1 << (7 - (i%8)));
        }
    }
}

void lcvdbt_repack_mtf7_proxy(const uint8_t *src, float scale, uint8_t *dst, int dim) {
    float *tmp = (float *)malloc((size_t)dim * sizeof(float));
    int plane_bytes = dim / 8;
    float inv_scale = scale > 0.0f ? (1.0f / scale) : 1.0f;
    for (int d = 0; d < dim; d++) {
        int byte_idx = d / 8;
        int bit_pos = 7 - (d % 8);
        int v = 0;
        int pow3 = 1;
        for (int i = 0; i < 7; i++) {
            if (src[i * plane_bytes + byte_idx] & (1 << bit_pos)) v += pow3;
            if (src[(i + 7) * plane_bytes + byte_idx] & (1 << bit_pos)) v -= pow3;
            pow3 *= 3;
        }
        tmp[d] = (float)v * inv_scale;
    }
    lcvdbt_pack_f32_bpmt7(tmp, dst, dim);
    free(tmp);
}

void lcvdbt_pack_mtf7_fingerprint(const uint8_t *src, uint8_t *dst, int dim) {
    int plane_bytes = dim / 8;
    int fp_bytes = (dim + 1) / 2;
    memset(dst, 0, (size_t)fp_bytes);
    for (int d = 0; d < dim; d++) {
        int byte_idx = d / 8;
        int bit_pos = 7 - (d % 8);
        int fp = 0;
        for (int i = 6; i >= 0; i--) {
            if (src[i * plane_bytes + byte_idx] & (1 << bit_pos)) { fp = i + 1; break; }
            if (src[(i + 7) * plane_bytes + byte_idx] & (1 << bit_pos)) { fp = -(i + 1); break; }
        }
        uint8_t code = (uint8_t)(fp + 8);
        if ((d & 1) == 0) dst[d >> 1] = (uint8_t)(code << 4);
        else dst[d >> 1] |= code;
    }
}

size_t lcvdbt_sizeof_topo(uint32_t max_nodes) { return (size_t)max_nodes * sizeof(lcvdbt_topo_t); }
size_t lcvdbt_sizeof_vec(uint32_t max_nodes, uint16_t dim, uint8_t quant) {
    uint16_t vec_bytes = LCVDBT_QUANT_VEC_BYTES(dim, quant);
    return (size_t)max_nodes * vec_bytes;
}
size_t lcvdbt_sizeof_vis(uint32_t max_nodes) { return max_nodes; }

lcvdbt_t *lcvdbt_new(uint32_t max_nodes, uint16_t dim, uint8_t quant) {
    if (dim == 0 || dim % 64 != 0) return NULL;
    if (max_nodes == 0) return NULL;
    if (quant < 1 || quant > 7) return NULL;
    size_t topo_size = lcvdbt_sizeof_topo(max_nodes);
    size_t vec_size = lcvdbt_sizeof_vec(max_nodes, dim, quant);
    size_t vis_size = lcvdbt_sizeof_vis(max_nodes);
    size_t topo_aligned = (topo_size + 63) & ~63;
    size_t vec_aligned = (vec_size + 63) & ~63;
    size_t vis_aligned = (vis_size + 63) & ~63;
    size_t total = sizeof(lcvdbt_t) + topo_aligned + vec_aligned + vis_aligned;
    void *mem = lcvdb_aligned_alloc(64, total);
    if (!mem) return NULL;
    lcvdbt_t *db = (lcvdbt_t *)mem;
    uint8_t *buf = (uint8_t *)mem + sizeof(lcvdbt_t);
    init_common(db, buf, buf + topo_aligned, buf + topo_aligned + vec_aligned, max_nodes, dim, quant);
    db->_managed = 1;
    return db;
}

void lcvdbt_free(lcvdbt_t *db) {
    if (!db) return;
    if (db->mtf_scales) free(db->mtf_scales);
    if (db->quant_norms) free(db->quant_norms);
    if (db->sign_array && db->quant == LCVDBT_QUANT_MTF7) free(db->sign_array);
    if (db->_managed) free(db);
}

/* ---------- BPMT7 (True Ternary Compute) Scalar Fallback ---------- */

/* End of init_ternary.c */
