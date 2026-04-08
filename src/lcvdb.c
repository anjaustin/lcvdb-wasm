/* lcvdb.c — Public API implementation.
 *
 * Wraps the internal engine types with a clean 7-function API.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include "platform.h"
#include "lcvdb_ternary.h"
#include "sce.h"
#include "../include/lcvdb.h"

struct lcvdb {
    lcvdbt_t *db;
    uint8_t  *trit_db;
    float    *quant_norms;
    uint16_t  dim;
    uint32_t  capacity;
    int       trit_bytes;
};

lcvdb_t *lcvdb_create(uint16_t dim, uint32_t capacity) {
    if (dim == 0 || dim % 64 != 0 || capacity == 0) return NULL;

    lcvdb_t *lc = calloc(1, sizeof(lcvdb_t));
    if (!lc) return NULL;

    lc->dim = dim;
    lc->capacity = capacity;
    lc->trit_bytes = ((dim / 4 + 31) / 32) * 32; /* 32-byte aligned */

    lc->db = lcvdbt_new(capacity, dim, LCVDBT_QUANT_MTF7);
    if (!lc->db) { free(lc); return NULL; }

    lc->trit_db = (uint8_t *)lcvdb_aligned_alloc(32, (size_t)capacity * lc->trit_bytes);
    if (!lc->trit_db) { lcvdbt_free(lc->db); free(lc); return NULL; }
    memset(lc->trit_db, 0, (size_t)capacity * lc->trit_bytes);

    lc->quant_norms = calloc(capacity, sizeof(float));
    if (!lc->quant_norms) { free(lc->trit_db); lcvdbt_free(lc->db); free(lc); return NULL; }

    return lc;
}

void lcvdb_free(lcvdb_t *lc) {
    if (!lc) return;
    lcvdbt_free(lc->db);
    free(lc->trit_db);
    free(lc->quant_norms);
    free(lc);
}

uint32_t lcvdb_insert(lcvdb_t *lc, uint32_t id, const float *vector) {
    if (!lc || !vector) return UINT32_MAX;
    if (lc->db->node_count >= lc->capacity) return UINT32_MAX; /* TODO: grow */

    uint16_t dim = lc->dim;
    uint16_t vb = lcvdbt_vec_bytes(lc->db);
    int fpb = (dim + 1) / 2;

    /* Pack float32 → MTF7 */
    uint8_t *packed = malloc(vb);
    if (!packed) return UINT32_MAX;
    float mtf_scale;
    lcvdbt_pack_f32_mtf7(vector, packed, dim, &mtf_scale);

    /* Insert into engine */
    uint32_t node_id = lcvdbt_insert_flat(lc->db, packed, id, mtf_scale);
    if (node_id == LCVDBT_INVALID_ID) { free(packed); return UINT32_MAX; }

    /* Encode trit fingerprint */
    uint8_t *fp = malloc(fpb);
    if (fp) {
        lcvdbt_pack_mtf7_fingerprint(packed, fp, dim);
        sce_encode_trit(fp, dim, lc->trit_db + (size_t)node_id * lc->trit_bytes);
        free(fp);
    }

    /* Compute norm */
    int64_t sd = lcvdbt_dot_bpmt7(packed, packed, dim);
    lc->quant_norms[node_id] = sd > 0 ? sqrtf((float)sd) : 0.0f;

    free(packed);
    return node_id;
}

int lcvdb_search(lcvdb_t *lc, const float *query, int k,
                 uint32_t *result_ids, float *result_scores) {
    if (!lc || !query || k <= 0) return 0;
    uint32_t n = lc->db->node_count;
    if (n == 0) return 0;

    return sce_search_parallel(
        query, lc->trit_db, lc->db->vec_array, lc->quant_norms,
        lc->db->topo_array, n, lc->dim,
        k, lc->trit_bytes, 1, NULL,
        result_ids, result_scores);
}

int lcvdb_save(const lcvdb_t *lc, const char *path) {
    if (!lc || !path) return -1;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    int rc = lcvdbt_save(lc->db, fd);
    close(fd);
    return rc;
}

lcvdb_t *lcvdb_load(const char *path) {
    if (!path) return NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    /* Read header to get dim and max_nodes */
    lcvdbt_t hdr;
    if (read(fd, &hdr, sizeof(hdr)) != sizeof(hdr)) { close(fd); return NULL; }
    if (hdr.M != LCVDBT_M || hdr.max_nodes == 0 || hdr.vec_dim == 0) { close(fd); return NULL; }

    lseek(fd, 0, SEEK_SET);

    lcvdb_t *lc = lcvdb_create(hdr.vec_dim, hdr.max_nodes);
    if (!lc) { close(fd); return NULL; }

    /* Save pointers that lcvdbt_new set up (header read will zero them) */
    lcvdbt_topo_t *saved_topo = lc->db->topo_array;
    uint8_t *saved_vec = lc->db->vec_array;
    uint8_t *saved_vis = lc->db->visited_buf;
    float *saved_scales = lc->db->mtf_scales;
    float *saved_norms = lc->db->quant_norms;

    /* Read header (overwrites db struct — pointers become NULL) */
    if (read(fd, lc->db, sizeof(lcvdbt_t)) != sizeof(lcvdbt_t)) { close(fd); lcvdb_free(lc); return NULL; }

    /* Restore pointers */
    lc->db->topo_array = saved_topo;
    lc->db->vec_array = saved_vec;
    lc->db->visited_buf = saved_vis;
    lc->db->mtf_scales = saved_scales;
    lc->db->quant_norms = saved_norms;
    lc->db->hooks = NULL;
    lc->db->visited_gen = 1;

    /* Read topo + vec arrays */
    size_t topo_size = (size_t)hdr.max_nodes * sizeof(lcvdbt_topo_t);
    uint16_t vb = LCVDBT_QUANT_VEC_BYTES(hdr.vec_dim, hdr.quant);
    if ((ssize_t)read(fd, lc->db->topo_array, topo_size) != (ssize_t)topo_size) { close(fd); lcvdb_free(lc); return NULL; }
    if ((ssize_t)read(fd, lc->db->vec_array, (size_t)hdr.max_nodes * vb) != (ssize_t)((size_t)hdr.max_nodes * vb)) { close(fd); lcvdb_free(lc); return NULL; }

    /* Read mtf_scales if save_fmt >= 1 */
    if (lc->db->save_fmt >= 1 && saved_scales) {
        (void)!read(fd, saved_scales, (size_t)hdr.max_nodes * sizeof(float));
    }

    close(fd);

    /* Rebuild trit fingerprints + norms from loaded vectors */
    int fpb = (lc->dim + 1) / 2;
    uint8_t *packed_buf = malloc(vb);
    uint8_t *fp_buf = malloc(fpb);
    for (uint32_t i = 0; i < lc->db->node_count; i++) {
        if (lc->db->topo_array[i].flags & LCVDBT_FLAG_DELETED) continue;
        const uint8_t *vec = lcvdbt_vec_at(lc->db, i);
        if (fp_buf) {
            lcvdbt_pack_mtf7_fingerprint(vec, fp_buf, lc->dim);
            sce_encode_trit(fp_buf, lc->dim, lc->trit_db + (size_t)i * lc->trit_bytes);
        }
        int64_t sd = lcvdbt_dot_bpmt7(vec, vec, lc->dim);
        lc->quant_norms[i] = sd > 0 ? sqrtf((float)sd) : 0.0f;
    }
    free(packed_buf);
    free(fp_buf);

    return lc;
}

uint32_t lcvdb_count(const lcvdb_t *lc) {
    return lc ? lc->db->node_count : 0;
}

void lcvdb_delete(lcvdb_t *lc, uint32_t id) {
    if (lc) lcvdbt_delete(lc->db, id);
}
