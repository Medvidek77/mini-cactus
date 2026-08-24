/*
 * needle.h - Public C API for Needle 2 Inference Engine
 * Production-ready suckless implementation.
 */

#ifndef NEEDLE_H
#define NEEDLE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Binary format constants */
#define NEEDLE_MAGIC 0x324C444E /* "NDL2" */
#define NEEDLE_MAX_WINDOW 256

typedef enum {
    NEEDLE_BACKEND_CPU = 0,
    NEEDLE_BACKEND_VULKAN = 1
} needle_backend_t;

/* Model Configuration & File Header */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t engram_vocab_size;
    uint32_t engram_dim;
    uint32_t max_seq_len;
    uint8_t  padding[84];
} needle_header_t;

typedef struct needle_context needle_context_t;

typedef struct {
    needle_backend_t backend;
    bool enable_grammar;
    const char *json_schema;
    float temperature;
    float top_p;
} needle_config_t;

#define NEEDLE_VOCAB_TOKEN_SIZE 32
#define NEEDLE_MAX_OUTPUT_TOKENS 512

typedef struct {
    float confidence;
    size_t generated_tokens;
    uint32_t token_ids[NEEDLE_MAX_OUTPUT_TOKENS];
} needle_output_meta_t;

/* Core API Functions */
needle_context_t *needle_open(const char *filepath, needle_config_t config);
int needle_eval(needle_context_t *ctx, const uint8_t *prompt, size_t prompt_len, uint8_t *out_buf, size_t max_out_len, needle_output_meta_t *meta);
const char *needle_get_token_str(needle_context_t *ctx, uint32_t token_id);
void needle_close(needle_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* NEEDLE_H */
