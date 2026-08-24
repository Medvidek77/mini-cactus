/*
 * tokenizer.h - Subword / Byte-level BPE Tokenizer for Needle 2
 */

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define TOKENIZER_MAX_VOCAB 8192
#define TOKENIZER_SLOT_SIZE 32

typedef struct {
    const char *vocab_table;
    size_t vocab_size;
} tokenizer_t;

void tokenizer_init(tokenizer_t *tok, const char *vocab_table, size_t vocab_size);
size_t tokenizer_encode(tokenizer_t *tok, const char *text, uint32_t *out_tokens, size_t max_tokens);
size_t tokenizer_decode(tokenizer_t *tok, const uint32_t *tokens, size_t num_tokens, char *out_text, size_t max_len);

#endif /* TOKENIZER_H */
