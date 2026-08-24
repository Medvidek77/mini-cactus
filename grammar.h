/*
 * grammar.h - Lightweight JSON Schema Grammar Constraint State Machine
 */

#ifndef GRAMMAR_H
#define GRAMMAR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    GRAMMAR_STATE_START,
    GRAMMAR_STATE_OBJECT_OPEN,
    GRAMMAR_STATE_KEY,
    GRAMMAR_STATE_COLON,
    GRAMMAR_STATE_VALUE,
    GRAMMAR_STATE_COMMA,
    GRAMMAR_STATE_OBJECT_CLOSE,
    GRAMMAR_STATE_END
} grammar_state_t;

typedef struct {
    grammar_state_t state;
    bool in_string;
    int depth;
    bool active;
} grammar_ctx_t;

void grammar_init(grammar_ctx_t *gctx, const char *schema_json);
bool grammar_accept(grammar_ctx_t *gctx, uint8_t byte);
void grammar_mask_logits(grammar_ctx_t *gctx, float *logits, size_t vocab_size);

#endif /* GRAMMAR_H */
