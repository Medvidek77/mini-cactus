/*
 * grammar.c - Lightweight JSON Schema Grammar Constraint State Machine
 */

#include "grammar.h"
#include <string.h>
#include <ctype.h>

void grammar_init(grammar_ctx_t *gctx, const char *schema_json) {
    if (!gctx) return;
    gctx->state = GRAMMAR_STATE_START;
    gctx->in_string = false;
    gctx->depth = 0;
    gctx->active = (schema_json != NULL && strlen(schema_json) > 0);
}

bool grammar_accept(grammar_ctx_t *gctx, uint8_t byte) {
    if (!gctx || !gctx->active) return true;

    if (gctx->in_string) {
        if (byte == '"') {
            gctx->in_string = false;
        }
        return true;
    }

    if (byte == '"') {
        gctx->in_string = true;
        return true;
    }

    switch (gctx->state) {
        case GRAMMAR_STATE_START:
            if (byte == '{') {
                gctx->state = GRAMMAR_STATE_KEY;
                gctx->depth++;
                return true;
            }
            return false;

        case GRAMMAR_STATE_KEY:
            if (byte == ':') {
                gctx->state = GRAMMAR_STATE_VALUE;
                return true;
            }
            return true;

        case GRAMMAR_STATE_VALUE:
            if (byte == ',') {
                gctx->state = GRAMMAR_STATE_KEY;
                return true;
            } else if (byte == '}') {
                gctx->depth--;
                if (gctx->depth == 0) gctx->state = GRAMMAR_STATE_END;
                return true;
            }
            return true;

        case GRAMMAR_STATE_END:
            return false;

        default:
            return true;
    }
}

void grammar_mask_logits(grammar_ctx_t *gctx, float *logits, size_t vocab_size) {
    if (!gctx || !gctx->active) return;

    for (size_t b = 0; b < vocab_size; b++) {
        grammar_ctx_t temp = *gctx;
        if (!grammar_accept(&temp, (uint8_t)b)) {
            logits[b] = -1e9f;
        }
    }
}
