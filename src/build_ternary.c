/* build_ternary.c — Flat insert/delete/update for SCE engine
 *
 * The HNSW graph construction path has been removed. The SCE engine
 * searches via flat trit Hamming scan — no graph is needed. Insert is
 * now O(1): pack + append. The old graph-building code is archived in
 * archived/pre-strip/build_ternary_pre_flat.c.
 */
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "lcvdb_ternary.h"

/* ── Flat Insert: O(1) pack + append ── */

uint32_t lcvdbt_insert_flat(lcvdbt_t *db, const uint8_t *packed_vec,
                             uint32_t payload_id, float mtf_scale) {
    LCVDBT_LOCK_WRITE(db);

    /* Allocate node ID: reuse from free list or bump node_count */
    uint32_t node_id;
    if (db->free_head != LCVDBT_INVALID_ID) {
        node_id = db->free_head;
        db->free_head = db->topo_array[node_id].payload_id; /* free list stored in payload_id */
        db->free_count--;
    } else {
        if (db->node_count >= db->max_nodes) {
            LCVDBT_UNLOCK(db);
            return LCVDBT_INVALID_ID;
        }
        node_id = db->node_count++;
    }

    /* Copy packed vector */
    uint16_t vb = lcvdbt_vec_bytes(db);
    memcpy(db->vec_array + (size_t)node_id * vb, packed_vec, vb);

    /* Set topology metadata (clear graph edges — not used by SCE) */
    memset(&db->topo_array[node_id], 0, sizeof(lcvdbt_topo_t));
    db->topo_array[node_id].payload_id = payload_id;

    /* Store MTF7 scale and norm */
    if (db->mtf_scales)
        db->mtf_scales[node_id] = mtf_scale;

    if (db->quant_norms) {
        int64_t sd = lcvdbt_dot_bpmt7(packed_vec, packed_vec, db->vec_dim);
        db->quant_norms[node_id] = sd > 0 ? sqrtf((float)sd) : 0.0f;
    }

    /* Update entry point if this is the first node */
    if (db->entry_point == LCVDBT_INVALID_ID)
        db->entry_point = node_id;

    LCVDBT_UNLOCK(db);
    return node_id;
}

/* ── Legacy insert wrappers (called by WAL replay) ── */

uint32_t lcvdbt_insert_mtf7(lcvdbt_t *db, const uint8_t *packed_vec,
                              uint32_t payload_id, float mtf_scale) {
    return lcvdbt_insert_flat(db, packed_vec, payload_id, mtf_scale);
}

uint32_t lcvdbt_insert(lcvdbt_t *db, const uint8_t *packed_vec,
                        uint32_t payload_id) {
    return lcvdbt_insert_flat(db, packed_vec, payload_id, 1.0f);
}

/* ── Delete: soft-delete with tombstone ── */

void lcvdbt_delete(lcvdbt_t *db, uint32_t node_id) {
    LCVDBT_LOCK_WRITE(db);
    if (node_id < db->node_count &&
        !(db->topo_array[node_id].flags & LCVDBT_FLAG_DELETED)) {
        db->topo_array[node_id].flags |= LCVDBT_FLAG_DELETED;
        db->topo_array[node_id].payload_id = db->free_head;
        db->free_head = node_id;
        db->free_count++;
    }
    LCVDBT_UNLOCK(db);
}

/* ── Update: overwrite packed vector in place ── */

void lcvdbt_update(lcvdbt_t *db, uint32_t node_id, const uint8_t *packed_vec) {
    LCVDBT_LOCK_WRITE(db);
    if (node_id < db->node_count) {
        uint16_t vb = lcvdbt_vec_bytes(db);
        memcpy(db->vec_array + (size_t)node_id * vb, packed_vec, vb);
    }
    LCVDBT_UNLOCK(db);
}
