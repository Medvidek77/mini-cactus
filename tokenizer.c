/*
 * tokenizer.c - Subword / Byte-level BPE Tokenizer for Needle 2
 */

#include "tokenizer.h"
#include <string.h>
#include <stdio.h>

void tokenizer_init(tokenizer_t *tok, const char *vocab_table, size_t vocab_size) {
    if (!tok) return;
    tok->vocab_table = vocab_table;
    tok->vocab_size = (vocab_size > TOKENIZER_MAX_VOCAB) ? TOKENIZER_MAX_VOCAB : vocab_size;
}

size_t tokenizer_encode(tokenizer_t *tok, const char *text, uint32_t *out_tokens, size_t max_tokens) {
    if (!tok || !text || !out_tokens || max_tokens == 0) return 0;

    size_t text_len = strlen(text);
    size_t pos = 0;
    size_t count = 0;

    while (pos < text_len && count < max_tokens) {
        size_t best_len = 0;
        uint32_t best_id = 0;

        /* Longest prefix match against vocabulary table */
        if (tok->vocab_table) {
            for (size_t id = 0; id < tok->vocab_size; id++) {
                const char *entry = &tok->vocab_table[id * TOKENIZER_SLOT_SIZE];
                size_t elen = strlen(entry);
                if (elen > 0 && elen <= (text_len - pos)) {
                    if (strncmp(&text[pos], entry, elen) == 0) {
                        if (elen > best_len) {
                            best_len = elen;
                            best_id = (uint32_t)id;
                        }
                    }
                }
            }
        }

        if (best_len > 0) {
            out_tokens[count++] = best_id;
            pos += best_len;
        } else {
            /* Single ASCII byte fallback token */
            out_tokens[count++] = (uint32_t)((uint8_t)text[pos]);
            pos++;
        }
    }

    return count;
}

size_t tokenizer_decode(tokenizer_t *tok, const uint32_t *tokens, size_t num_tokens, char *out_text, size_t max_len) {
    if (!tok || !tokens || !out_text || max_len == 0) return 0;

    size_t pos = 0;
    out_text[0] = '\0';

    for (size_t i = 0; i < num_tokens && pos + 1 < max_len; i++) {
        uint32_t tid = tokens[i];
        if (tok->vocab_table && tid < tok->vocab_size) {
            const char *entry = &tok->vocab_table[tid * TOKENIZER_SLOT_SIZE];
            size_t elen = strlen(entry);
            if (elen > 0 && pos + elen < max_len) {
                memcpy(&out_text[pos], entry, elen);
                pos += elen;
            }
        } else if (tid < 256 && pos + 1 < max_len) {
            out_text[pos++] = (char)tid;
        }
    }

    out_text[pos] = '\0';
    return pos;
}
