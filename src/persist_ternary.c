/* L-Cache VDB — Ternary Persistence (save/load via file descriptors)
 *
 * File format (all little-endian, naturally aligned):
 *   [0..127]                           lcvdbt_t header (128 bytes, pointers zeroed)
 *   [128..128+max_nodes*128)           topo_array (flat, 128 bytes per node)
 *   [128+max_nodes*128..]              vec_array  (flat, vec_bytes per slot)
 *
 * The vec_dim field at [10..11] in the header encodes the dimension.
 * On load, the dimension is validated and vec_bytes derived from it.
 *
 * On save: pointer fields (topo_array, vec_array, sign_array, visited_buf) are
 *          written as zero.
 * On load: pointer fields are restored to caller-provided buffers.
 *          The visited buffer is zeroed and generation reset to 1.
 * No endian conversion — assumes same architecture for save and load.
 */
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/mman.h>
#include "lcvdb_ternary.h"

/* Write exactly `len` bytes, handling partial writes. */
static int write_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n <= 0) return -1;
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

/* Read exactly `len` bytes, handling partial reads. */
static int read_all(int fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    while (len > 0) {
        ssize_t n = read(fd, p, len);
        if (n <= 0) return -1;
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

int lcvdbt_save(const lcvdbt_t *db, int fd) {
    /* Write header with pointers zeroed, save_fmt=1 (includes mtf_scales) */
    lcvdbt_t header;
    memcpy(&header, db, sizeof(lcvdbt_t));
    header.topo_array = NULL;
    header.vec_array = NULL;
    header.sign_array = NULL;
    header.visited_buf = NULL;
    header.visited_gen = 0;
    header.mtf_scales = NULL;
    header.quant_norms = NULL;
    header.hooks = NULL;
    header.save_fmt = 1; /* format version: includes mtf_scales */

    if (write_all(fd, &header, sizeof(lcvdbt_t)) != 0)
        return -1;

    /* Write topo array */
    size_t topo_size = (size_t)db->max_nodes * sizeof(lcvdbt_topo_t);
    if (write_all(fd, db->topo_array, topo_size) != 0)
        return -1;

    /* Write vec array (vec_bytes per node) */
    uint16_t vb = lcvdbt_vec_bytes(db);
    size_t vec_size = (size_t)db->max_nodes * vb;
    if (write_all(fd, db->vec_array, vec_size) != 0)
        return -1;

    /* Write mtf_scales (save_fmt >= 1) */
    {
        size_t scales_size = (size_t)db->max_nodes * sizeof(float);
        if (db->mtf_scales) {
            if (write_all(fd, db->mtf_scales, scales_size) != 0)
                return -1;
        } else {
            void *zeros = calloc(db->max_nodes, sizeof(float));
            if (!zeros) return -1;
            int rc = write_all(fd, zeros, scales_size);
            free(zeros);
            if (rc != 0) return -1;
        }
    }

    return 0;
}

int lcvdbt_load(lcvdbt_t *db, int fd, void *topo_buf, void *vec_buf, void *vis_buf) {
    /* Read header */
    if (read_all(fd, db, sizeof(lcvdbt_t)) != 0)
        return -1;

    /* Validate basic fields */
    if (db->M != LCVDBT_M || db->max_nodes == 0)
        return -1;

    /* Validate dimension: must be > 0, multiple of 64 */
    if (db->vec_dim == 0 || (db->vec_dim % 64) != 0)
        return -1;

    /* Validate quant level: 1-7 are valid.
     * Legacy files with quant == 0 are treated as ternary. */
    if (db->quant == 0) db->quant = LCVDBT_QUANT_TERNARY;
    if (db->quant < 1 || db->quant > 7)
        return -1;

    /* Restore pointers to caller-provided buffers */
    db->topo_array = (lcvdbt_topo_t *)topo_buf;
    db->vec_array = (uint8_t *)vec_buf;
    db->visited_buf = (uint8_t *)vis_buf;
    db->visited_gen = 1;

    /* Read topo array */
    size_t topo_size = (size_t)db->max_nodes * sizeof(lcvdbt_topo_t);
    if (read_all(fd, topo_buf, topo_size) != 0)
        return -1;

    /* Read vec array */
    uint16_t vb = lcvdbt_vec_bytes(db);
    size_t vec_size = (size_t)db->max_nodes * vb;
    if (read_all(fd, vec_buf, vec_size) != 0)
        return -1;

    /* Read mtf_scales if save_fmt >= 1 */
    if (db->save_fmt >= 1) {
        if (!db->mtf_scales)
            db->mtf_scales = calloc(db->max_nodes, sizeof(float));
        if (db->mtf_scales) {
            size_t scales_size = (size_t)db->max_nodes * sizeof(float);
            if (read_all(fd, db->mtf_scales, scales_size) != 0)
                return -1;
        }
    }

    /* Clear visited buffer */
    memset(vis_buf, 0, db->max_nodes);

    return 0;
}

/* ==========================================================================
 * MMAP Persistence — read-only memory-mapped access.
 * ========================================================================== */

void *lcvdbt_mmap_load(int fd, lcvdbt_t **db_out, void *vis_buf, size_t *file_size_out) {
    /* Get file size */
    off_t off = lseek(fd, 0, SEEK_END);
    if (off < 0) return NULL;
    size_t file_size = (size_t)off;
    if (file_size < sizeof(lcvdbt_t)) return NULL;
    
    /* Map the entire file with PROT_READ | PROT_WRITE and MAP_PRIVATE.
     * MAP_PRIVATE gives copy-on-write semantics, so our pointer patching
     * won't modify the underlying file. The file itself remains read-only. */
    void *base = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) return NULL;
    
    /* The header is at offset 0 */
    lcvdbt_t *db = (lcvdbt_t *)base;
    
    /* Validate basic fields */
    if (db->M != LCVDBT_M || db->max_nodes == 0) {
        munmap(base, file_size);
        return NULL;
    }
    
    /* Validate dimension: must be > 0, multiple of 64 */
    if (db->vec_dim == 0 || (db->vec_dim % 64) != 0) {
        munmap(base, file_size);
        return NULL;
    }
    
    /* Validate quant level: 1-7 are valid.
     * Legacy files with quant == 0 are treated as ternary. */
    if (db->quant == 0) db->quant = LCVDBT_QUANT_TERNARY;
    if (db->quant < 1 || db->quant > 7) {
        munmap(base, file_size);
        return NULL;
    }
    
    /* Validate file size matches expected (legacy or with scales) */
    uint16_t vb = LCVDBT_QUANT_VEC_BYTES(db->vec_dim, db->quant);
    size_t base_size = sizeof(lcvdbt_t) + (size_t)db->max_nodes * sizeof(lcvdbt_topo_t) + (size_t)db->max_nodes * vb;
    if (file_size < base_size) {
        munmap(base, file_size);
        return NULL;
    }
    
    /* The mmap'd file has NULL pointers for topo_array and vec_array.
     * We need to patch them to point into the mmap. This is safe because
     * we're using MAP_PRIVATE (copy-on-write), so writes to the mapped
     * region won't affect the underlying file. */
    
    /* Calculate offsets */
    uint8_t *bytes = (uint8_t *)base;
    db->topo_array = (lcvdbt_topo_t *)(bytes + sizeof(lcvdbt_t));
    db->vec_array = bytes + sizeof(lcvdbt_t) + (size_t)db->max_nodes * sizeof(lcvdbt_topo_t);
    
    /* Visited buffer is caller-provided (not in mmap) */
    db->visited_buf = (uint8_t *)vis_buf;
    db->visited_gen = 1;
    memset(vis_buf, 0, db->max_nodes);
    
    /* Copy mtf_scales if save_fmt >= 1 and data is present.
     * We copy rather than point into the mmap so lcvdbt_free() can safely free(). */
    if (db->save_fmt >= 1) {
        size_t scales_offset = sizeof(lcvdbt_t) +
            (size_t)db->max_nodes * sizeof(lcvdbt_topo_t) +
            (size_t)db->max_nodes * vb;
        size_t scales_size = (size_t)db->max_nodes * sizeof(float);
        if (file_size >= scales_offset + scales_size) {
            db->mtf_scales = malloc(scales_size);
            if (db->mtf_scales)
                memcpy(db->mtf_scales, bytes + scales_offset, scales_size);
        } else {
            db->mtf_scales = NULL;
        }
    } else {
        db->mtf_scales = NULL;
    }

    /* Clear mutable fields that shouldn't persist */
    db->hooks = NULL;
    db->quant_norms = NULL;
    db->sign_array = NULL;
    
    if (file_size_out) *file_size_out = file_size;
    if (db_out) *db_out = db;
    
    return base;
}

int lcvdbt_mmap_unload(void *mmap_base, size_t file_size) {
    if (!mmap_base) return -1;
    return munmap(mmap_base, file_size);
}
