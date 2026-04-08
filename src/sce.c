/* sce.c — Spatial Compute Engine
 *
 * Fused scan-rerank kernel. Three stages:
 *   Stage 1: Encode query (float → MTF7 → trit)     ~25 μs
 *   Stage 2: Trit Hamming scan with bounded heap     O(N) bandwidth
 *   Stage 3: Dot-product rerank on shortlist          O(50) dot products
 *
 * Zero dependencies beyond lcvdb_ternary.h and sce.h.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef __EMSCRIPTEN__
#include <pthread.h>
#endif
#include "platform.h"
#include "sce.h"
#include "lcvdb_ternary.h"

int sce_search(
    const float   *query,
    const uint8_t *trit_db,
    const uint8_t *vec_db,
    const float   *norms,
    uint32_t       n,
    uint16_t       dim,
    int            k,
    int            trit_bytes,
    uint32_t      *result_ids,
    float         *result_scores,
    const uint32_t *inject_ids,
    int             inject_count)
{
    if (n == 0 || k <= 0) return 0;

    int vb = LCVDBT_QUANT_VEC_BYTES(dim, LCVDBT_QUANT_MTF7);
    int fpb = (dim + 1) / 2;

    /* ── Stage 1: Encode query ── */
    int qtrit_size = ((dim / 4 + 31) / 32) * 32; /* match trit_bytes alignment */
    uint8_t *qpacked = (uint8_t *)malloc(vb);
    uint8_t *qfp = (uint8_t *)malloc(fpb);
    uint8_t *qtrit = NULL;
    if (!qpacked || !qfp || (qtrit = (uint8_t *)lcvdb_aligned_alloc(32, qtrit_size)) == NULL) {
        free(qpacked); free(qfp); free(qtrit); return 0;
    }
    memset(qtrit, 0, qtrit_size);
    float qscale;

    lcvdbt_pack_f32_mtf7(query, qpacked, dim, &qscale);
    lcvdbt_pack_mtf7_fingerprint(qpacked, qfp, dim);
    sce_encode_trit(qfp, dim, qtrit);

    int64_t qsd = lcvdbt_dot_bpmt7(qpacked, qpacked, dim);
    float qnorm = qsd > 0 ? sqrtf((float)qsd) : 0.0f;
    if (qnorm <= 0) { free(qpacked); free(qfp); free(qtrit); return 0; }

    /* ── Stage 2: Trit Hamming scan with bounded heap ── */
    int sl = SCE_SHORTLIST;
    sce_cand_t heap[SCE_SHORTLIST];
    int hc = 0;

    for (uint32_t i = 0; i < n; i++) {
        /* Prefetch 4 vectors ahead */
        if (i + 4 < n)
            lcvdb_prefetch(trit_db + (size_t)(i + 4) * trit_bytes, 0, 0);

        int32_t ham = (trit_bytes == SCE_TRIT_BYTES_768)
                    ? sce_hamming192(qtrit, trit_db + (size_t)i * trit_bytes)
                    : sce_hamming64(qtrit, trit_db + (size_t)i * trit_bytes);

        if (hc < sl) {
            heap[hc].ham = ham;
            heap[hc].id = i;
            hc++;
            if (hc == sl) sce_heap_build(heap, hc);
        } else if (ham < heap[0].ham) {
            heap[0].ham = ham;
            heap[0].id = i;
            sce_heap_sift(heap, hc, 0);
        }
    }

    /* ── Merge injected candidates (from CfC router or codebook) ── */
    int total_cands = hc;
    uint32_t cand_ids[SCE_SHORTLIST + 256]; /* heap + injected */
    for (int i = 0; i < hc; i++) cand_ids[i] = heap[i].id;

    if (inject_ids && inject_count > 0) {
        /* Deduplicate injected against heap */
        for (int i = 0; i < inject_count && total_cands < SCE_SHORTLIST + 256; i++) {
            uint32_t id = inject_ids[i];
            int dup = 0;
            for (int j = 0; j < total_cands; j++) {
                if (cand_ids[j] == id) { dup = 1; break; }
            }
            if (!dup) cand_ids[total_cands++] = id;
        }
    }

    /* ── Stage 3: Dot-product rerank ── */
    typedef struct { float score; uint32_t id; } result_t;
    result_t top[SCE_SHORTLIST + 256];
    int tc = 0;

    for (int i = 0; i < total_cands; i++) {
        uint32_t id = cand_ids[i];
        if (id >= n) continue;
        float vn = norms[id];
        if (vn <= 0) continue;

        const uint8_t *vec = vec_db + (size_t)id * vb;
        int64_t d = lcvdbt_dot_bpmt7(qpacked, vec, dim);
        float score = (float)((double)d / ((double)qnorm * (double)vn));

        if (tc < k) {
            top[tc].score = score;
            top[tc].id = id;
            tc++;
        } else {
            /* Replace minimum */
            int mi = 0;
            for (int j = 1; j < k; j++)
                if (top[j].score < top[mi].score) mi = j;
            if (score > top[mi].score) {
                top[mi].score = score;
                top[mi].id = id;
            }
        }
    }

    /* Sort results by score descending */
    for (int i = 0; i < tc; i++) {
        for (int j = i + 1; j < tc; j++) {
            if (top[j].score > top[i].score) {
                result_t t = top[i];
                top[i] = top[j];
                top[j] = t;
            }
        }
    }

    /* Output */
    int out = tc < k ? tc : k;
    for (int i = 0; i < out; i++) {
        result_ids[i] = top[i].id;
        result_scores[i] = top[i].score;
    }

    free(qpacked); free(qfp); free(qtrit);
    return out;
}

/* ── Production Parallel SCE: Multi-threaded AVX2 Scan ── */

#define SCE_LOCAL_SHORTLIST 200 /* Over-sample to prevent eviction leakage */

typedef struct {
    const uint8_t *trit_db;
    const uint8_t *qtrit;
    const lcvdbt_topo_t *topo;
    uint32_t start;
    uint32_t end;
    int trit_bytes;
    sce_filter_t filter;
    sce_cand_t shortlist[SCE_LOCAL_SHORTLIST];
    int count;
    uint8_t _pad[128];
} sce_thread_args_t;

static void *sce_worker(void *arg) {
    sce_thread_args_t *a = (sce_thread_args_t *)arg;
    int sl = SCE_LOCAL_SHORTLIST;
    a->count = 0;

    for (uint32_t i = a->start; i < a->end; i++) {
        /* Pre-filter: skip deleted and non-matching vectors */
        if (a->topo) {
            const lcvdbt_topo_t *t = &a->topo[i];
            if (t->flags & LCVDBT_FLAG_DELETED) continue;
            if (a->filter.tag && t->filter_tag != a->filter.tag) continue;
            if (a->filter.bits_lo && !(t->filter_bits_lo & a->filter.bits_lo)) continue;
            if (a->filter.bits_hi && !(t->filter_bits_hi & a->filter.bits_hi)) continue;
            if (a->filter.has_range) {
                if (t->filter_min < a->filter.range_min) continue;
                if (t->filter_max > a->filter.range_max) continue;
            }
        }

        if (i + 16 < a->end)
            lcvdb_prefetch(a->trit_db + (size_t)(i + 16) * a->trit_bytes, 0, 0);


        int32_t ham = (a->trit_bytes == SCE_TRIT_BYTES_768)
                    ? sce_hamming192(a->qtrit, a->trit_db + (size_t)i * a->trit_bytes)
                    : sce_hamming64(a->qtrit, a->trit_db + (size_t)i * a->trit_bytes);

        if (a->count < sl) {
            a->shortlist[a->count].ham = ham;
            a->shortlist[a->count].id = i;
            a->count++;
            if (a->count == sl) sce_heap_build(a->shortlist, sl);
        } else if (ham < a->shortlist[0].ham) {
            a->shortlist[0].ham = ham;
            a->shortlist[0].id = i;
            sce_heap_sift(a->shortlist, sl, 0);
        }
    }
    return NULL;
}

int sce_search_parallel(
    const float   *query,
    const uint8_t *trit_db,
    const uint8_t *vec_db,
    const float   *norms,
    const lcvdbt_topo_t *topo,
    uint32_t       n,
    uint16_t       dim,
    int            k,
    int            trit_bytes,
    int            num_threads,
    const sce_filter_t *filter,
    uint32_t      *result_ids,
    float         *result_scores)
{
    if (n == 0 || k <= 0) return 0;
#ifdef __EMSCRIPTEN__
    num_threads = 1; /* WASM: single-threaded */
#endif
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 64) num_threads = 64;

    /* Stage 1: Encode query (heap-allocated for 32-byte alignment on all libc) */
    int vb = LCVDBT_QUANT_VEC_BYTES(dim, LCVDBT_QUANT_MTF7);
    int fpb = (dim + 1) / 2;
    int qtrit_size = ((dim / 4 + 31) / 32) * 32;
    uint8_t *qpacked = (uint8_t *)malloc(vb);
    uint8_t *qfp = (uint8_t *)malloc(fpb);
    uint8_t *qtrit = NULL;
    if ((qtrit = (uint8_t *)lcvdb_aligned_alloc(32, qtrit_size)) == NULL) {
        free(qpacked); free(qfp); return 0;
    }
    memset(qtrit, 0, qtrit_size);
    float qscale;

    lcvdbt_pack_f32_mtf7(query, qpacked, dim, &qscale);
    lcvdbt_pack_mtf7_fingerprint(qpacked, qfp, dim);
    sce_encode_trit(qfp, dim, qtrit);

    int64_t qsd = lcvdbt_dot_bpmt7(qpacked, qpacked, dim);
    float qnorm = qsd > 0 ? sqrtf((float)qsd) : 0.0f;
    if (qnorm <= 0) { free(qpacked); free(qfp); free(qtrit); return 0; }

    /* Single-thread optimization: avoid pthread overhead for small tc */
    if (num_threads == 1) {
        sce_thread_args_t arg;
        arg.trit_db = trit_db; arg.qtrit = qtrit; arg.trit_bytes = trit_bytes;
        arg.start = 0; arg.end = n; arg.topo = topo;
        if (filter) arg.filter = *filter; else memset(&arg.filter, 0, sizeof(sce_filter_t));
        sce_worker(&arg);
        
        /* Local rerank and output */
        typedef struct { float score; uint32_t id; } result_t;
        result_t top[SCE_LOCAL_SHORTLIST];
        int tc = 0;
        for (int i = 0; i < arg.count; i++) {
            uint32_t id = arg.shortlist[i].id;
            float vn = norms[id]; if (vn <= 0) continue;
            int64_t d = lcvdbt_dot_bpmt7(qpacked, vec_db + (size_t)id * vb, dim);
            float score = (float)((double)d / ((double)qnorm * (double)vn));
            if (tc < k) {
                top[tc].score = score; top[tc].id = id; tc++;
            } else {
                int mi = 0; for (int j = 1; j < k; j++) if (top[j].score < top[mi].score || (top[j].score == top[mi].score && top[j].id > top[mi].id)) mi = j;
                if (score > top[mi].score || (score == top[mi].score && id < top[mi].id)) {
                    top[mi].score = score; top[mi].id = id;
                }
            }
        }
        for (int i = 0; i < tc; i++) {
            for (int j = i + 1; j < tc; j++) {
                if (top[j].score > top[i].score || (top[j].score == top[i].score && top[j].id < top[i].id)) {
                    result_t t = top[i]; top[i] = top[j]; top[j] = t;
                }
            }
        }
        int out = tc < k ? tc : k;
        for (int i = 0; i < out; i++) { result_ids[i] = top[i].id; result_scores[i] = top[i].score; }
        free(qpacked); free(qfp); free(qtrit);
        return out;
    }

    /* Stage 2: Parallel Hamming scan */
    pthread_t threads[64];
    sce_thread_args_t args[64];
    uint32_t chunk = (n + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; t++) {
        args[t].trit_db = trit_db;
        args[t].qtrit = qtrit;
        args[t].topo = topo;
        if (filter) args[t].filter = *filter; else memset(&args[t].filter, 0, sizeof(sce_filter_t));
        args[t].trit_bytes = trit_bytes;
        args[t].start = t * chunk;
        args[t].end = args[t].start + chunk;
        if (args[t].end > n) args[t].end = n;
        args[t].count = 0;
        if (args[t].start < n) {
            pthread_create(&threads[t], NULL, sce_worker, &args[t]);
        }
    }

    /* Wait and merge results from all threads */
    uint32_t cand_ids[64 * SCE_LOCAL_SHORTLIST];
    int total_cands = 0;

    for (int t = 0; t < num_threads; t++) {
        if (t * chunk < n) {
            pthread_join(threads[t], NULL);
            for (int i = 0; i < args[t].count; i++) {
                cand_ids[total_cands++] = args[t].shortlist[i].id;
            }
        }
    }

    /* Stage 3: Dot-product rerank on merged candidates */
    typedef struct { float score; uint32_t id; } result_t;
    result_t top[64 * SCE_LOCAL_SHORTLIST];
    int tc = 0;

    for (int i = 0; i < total_cands; i++) {
        uint32_t id = cand_ids[i];
        float vn = norms[id];
        if (vn <= 0) continue;
        int64_t d = lcvdbt_dot_bpmt7(qpacked, vec_db + (size_t)id * vb, dim);
        float score = (float)((double)d / ((double)qnorm * (double)vn));

        if (tc < k) {
            top[tc].score = score; top[tc].id = id; tc++;
        } else {
            /* Replace minimum */
            int mi = 0;
            for (int j = 1; j < tc; j++) {
                if (top[j].score < top[mi].score || (top[j].score == top[mi].score && top[j].id > top[mi].id)) mi = j;
            }
            if (score > top[mi].score || (score == top[mi].score && id < top[mi].id)) {
                top[mi].score = score; top[mi].id = id;
            }
        }
    }

    /* Final sort (Stable Tie-breaker) */
    for (int i = 0; i < tc; i++) {
        for (int j = i + 1; j < tc; j++) {
            if (top[j].score > top[i].score || (top[j].score == top[i].score && top[j].id < top[i].id)) {
                result_t t = top[i]; top[i] = top[j]; top[j] = t;
            }
        }
    }

    int out = tc < k ? tc : k;
    for (int i = 0; i < out; i++) {
        result_ids[i] = top[i].id;
        result_scores[i] = top[i].score;
    }
    free(qpacked); free(qfp); free(qtrit);
    return out;
}

/* ── Bulk trit encoding for database vectors ── */

void sce_encode_database(
    const float   *vectors,
    uint32_t       n,
    uint16_t       dim,
    uint8_t       *trit_out,
    int            trit_bytes,
    uint8_t       *vec_out,
    float         *norms_out)
{
    int vb = LCVDBT_QUANT_VEC_BYTES(dim, LCVDBT_QUANT_MTF7);
    int fpb = (dim + 1) / 2;
    uint8_t packed[vb];
    uint8_t fp[fpb];

    for (uint32_t i = 0; i < n; i++) {
        float scale;
        lcvdbt_pack_f32_mtf7(vectors + (size_t)i * dim, packed, dim, &scale);

        /* Store MTF7 vector for rerank */
        memcpy(vec_out + (size_t)i * vb, packed, vb);

        /* Extract fingerprint and encode trit */
        lcvdbt_pack_mtf7_fingerprint(packed, fp, dim);
        sce_encode_trit(fp, dim, trit_out + (size_t)i * trit_bytes);

        /* Compute norm */
        int64_t sd = lcvdbt_dot_bpmt7(packed, packed, dim);
        norms_out[i] = sd > 0 ? sqrtf((float)sd) : 0.0f;
    }
}
