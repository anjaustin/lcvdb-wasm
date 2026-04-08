/* lcvdb.h — Public API for the LCVDB embeddable vector search library.
 *
 * The smallest vector search engine. 42KB. Zero configuration. 100% recall.
 *
 * Thread safety: NOT thread-safe. External synchronization required
 * for concurrent access from multiple threads.
 *
 * Usage:
 *   lcvdb_t *db = lcvdb_create(768, 100000);
 *   lcvdb_insert(db, id, float_vector);
 *   lcvdb_search(db, query_vector, 10, result_ids, result_scores);
 *   lcvdb_save(db, "vectors.db");
 *   lcvdb_free(db);
 */

#ifndef LCVDB_H
#define LCVDB_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct lcvdb lcvdb_t;

/* Create a new database.
 * dim: vector dimensions (must be a multiple of 64)
 * capacity: initial capacity (doubles automatically when full) */
lcvdb_t *lcvdb_create(uint16_t dim, uint32_t capacity);

/* Free all memory. */
void lcvdb_free(lcvdb_t *db);

/* Insert a float32 vector. Returns internal node ID, or UINT32_MAX on failure.
 * The vector is packed to MTF7 ternary format internally. */
uint32_t lcvdb_insert(lcvdb_t *db, uint32_t id, const float *vector);

/* Search for the k nearest neighbors of a float32 query vector.
 * Returns the number of results found (≤ k).
 * result_ids and result_scores must have room for k entries. */
int lcvdb_search(lcvdb_t *db, const float *query, int k,
                 uint32_t *result_ids, float *result_scores);

/* Save database to a file. Returns 0 on success. */
int lcvdb_save(const lcvdb_t *db, const char *path);

/* Load database from a file. Returns NULL on failure. */
lcvdb_t *lcvdb_load(const char *path);

/* Get the number of vectors in the database. */
uint32_t lcvdb_count(const lcvdb_t *db);

/* Delete a vector by external ID (the ID passed to lcvdb_insert). */
void lcvdb_delete(lcvdb_t *db, uint32_t id);

#ifdef __cplusplus
}
#endif

#endif /* LCVDB_H */
