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
    uint32_t *id_map;       /* external_id → node_id hash table (size = capacity * 2) */
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

    size_t map_size = (size_t)capacity * 2;
    lc->id_map = malloc(map_size * sizeof(uint32_t));
    if (!lc->id_map) { free(lc->quant_norms); free(lc->trit_db); lcvdbt_free(lc->db); free(lc); return NULL; }
    for (size_t i = 0; i < map_size; i++) lc->id_map[i] = LCVDBT_INVALID_ID;

    return lc;
}

void lcvdb_free(lcvdb_t *lc) {
    if (!lc) return;
    lcvdbt_free(lc->db);
    free(lc->trit_db);
    free(lc->quant_norms);
    free(lc->id_map);
    free(lc);
}

/* ── ID map helpers (external_id ↔ node_id) ── */

static uint32_t id_hash(uint32_t id, uint32_t size) {
    id = ((id >> 16) ^ id) * 0x45d9f3b;
    id = ((id >> 16) ^ id) * 0x45d9f3b;
    id = (id >> 16) ^ id;
    return id % size;
}

static uint32_t lcvdb_map_get(const lcvdb_t *lc, uint32_t ext_id) {
    if (!lc->id_map) return LCVDBT_INVALID_ID;
    uint32_t size = lc->capacity * 2;
    uint32_t h = id_hash(ext_id, size);
    uint32_t start = h;
    while (lc->id_map[h] != LCVDBT_INVALID_ID) {
        uint32_t nid = lc->id_map[h];
        if (nid < lc->db->node_count &&
            lc->db->topo_array[nid].payload_id == ext_id &&
            !(lc->db->topo_array[nid].flags & LCVDBT_FLAG_DELETED))
            return nid;
        h = (h + 1) % size;
        if (h == start) break;
    }
    return LCVDBT_INVALID_ID;
}

static void lcvdb_map_insert(lcvdb_t *lc, uint32_t ext_id, uint32_t node_id) {
    if (!lc->id_map) return;
    uint32_t size = lc->capacity * 2;
    uint32_t h = id_hash(ext_id, size);
    while (lc->id_map[h] != LCVDBT_INVALID_ID) {
        if (lc->id_map[h] == node_id) return;
        h = (h + 1) % size;
    }
    lc->id_map[h] = node_id;
}

static int lcvdb_grow(lcvdb_t *lc) {
    uint32_t old_cap = lc->capacity;
    uint32_t new_cap = old_cap * 2;
    if (new_cap < old_cap) return -1; /* overflow */

    lcvdbt_t *new_db = lcvdbt_new(new_cap, lc->dim, LCVDBT_QUANT_MTF7);
    if (!new_db) return -1;

    /* Copy engine data */
    uint16_t vb = lcvdbt_vec_bytes(lc->db);
    uint32_t nc = lc->db->node_count;
    memcpy(new_db->topo_array, lc->db->topo_array, (size_t)nc * sizeof(lcvdbt_topo_t));
    memcpy(new_db->vec_array, lc->db->vec_array, (size_t)nc * vb);
    new_db->node_count = nc;
    new_db->entry_point = lc->db->entry_point;
    new_db->free_head = lc->db->free_head;
    new_db->free_count = lc->db->free_count;
    new_db->prng_state = lc->db->prng_state;
    if (lc->db->mtf_scales && new_db->mtf_scales)
        memcpy(new_db->mtf_scales, lc->db->mtf_scales, (size_t)nc * sizeof(float));
    if (lc->db->quant_norms && new_db->quant_norms)
        memcpy(new_db->quant_norms, lc->db->quant_norms, (size_t)nc * sizeof(float));

    lcvdbt_free(lc->db);
    lc->db = new_db;

    /* Grow trit_db */
    uint8_t *new_trit = (uint8_t *)lcvdb_aligned_alloc(32, (size_t)new_cap * lc->trit_bytes);
    if (!new_trit) return -1;
    memcpy(new_trit, lc->trit_db, (size_t)old_cap * lc->trit_bytes);
    memset(new_trit + (size_t)old_cap * lc->trit_bytes, 0, (size_t)(new_cap - old_cap) * lc->trit_bytes);
    free(lc->trit_db);
    lc->trit_db = new_trit;

    /* Grow quant_norms */
    float *new_norms = realloc(lc->quant_norms, (size_t)new_cap * sizeof(float));
    if (!new_norms) return -1;
    memset(new_norms + old_cap, 0, (size_t)(new_cap - old_cap) * sizeof(float));
    lc->quant_norms = new_norms;

    /* Grow id_map */
    uint32_t *new_map = realloc(lc->id_map, (size_t)new_cap * 2 * sizeof(uint32_t));
    if (!new_map) return -1;
    lc->id_map = new_map;
    /* Rebuild id_map */
    for (size_t i = 0; i < (size_t)new_cap * 2; i++) lc->id_map[i] = LCVDBT_INVALID_ID;
    for (uint32_t i = 0; i < nc; i++) {
        if (lc->db->topo_array[i].flags & LCVDBT_FLAG_DELETED) continue;
        uint32_t ext = lc->db->topo_array[i].payload_id;
        uint32_t h = ((ext >> 16) ^ ext) * 0x45d9f3b;
        h = ((h >> 16) ^ h) * 0x45d9f3b;
        h = ((h >> 16) ^ h) % (new_cap * 2);
        while (new_map[h] != LCVDBT_INVALID_ID) h = (h + 1) % (new_cap * 2);
        new_map[h] = i;
    }

    lc->capacity = new_cap;
    return 0;
}

uint32_t lcvdb_insert(lcvdb_t *lc, uint32_t id, const float *vector) {
    if (!lc || !vector) return UINT32_MAX;

    /* Grow if at capacity */
    if (lc->db->node_count >= lc->capacity) {
        if (lcvdb_grow(lc) != 0) return UINT32_MAX;
    }

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

    /* Register in id_map */
    lcvdb_map_insert(lc, id, node_id);

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

    /* Rebuild trit fingerprints + norms + id_map from loaded vectors */
    int fpb = (lc->dim + 1) / 2;
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
        lcvdb_map_insert(lc, lc->db->topo_array[i].payload_id, i);
    }
    free(fp_buf);

    return lc;
}

uint32_t lcvdb_count(const lcvdb_t *lc) {
    return lc ? lc->db->node_count : 0;
}

void lcvdb_delete(lcvdb_t *lc, uint32_t id) {
    if (!lc) return;
    uint32_t node_id = lcvdb_map_get(lc, id);
    if (node_id != LCVDBT_INVALID_ID)
        lcvdbt_delete(lc->db, node_id);
}
