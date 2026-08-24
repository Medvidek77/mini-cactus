/*
 * main.c - Suckless CLI Frontend for Needle 2
 */

#include "needle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --model <path>      Path to needle2.bin model (default: needle2.bin)\n");
    printf("  --prompt <text>     Input prompt text\n");
    printf("  --backend <cpu|vulkan> Backend compute engine (default: cpu)\n");
    printf("  --json-schema <str> Enforce JSON schema grammar\n");
    printf("  --help              Show this help menu\n");
}

int main(int argc, char **argv) {
    const char *model_path = "needle2.bin";
    const char *prompt_text = "Hello Needle 2";
    const char *schema = NULL;
    float temperature = 0.0f;
    needle_backend_t backend = NEEDLE_BACKEND_CPU;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "vulkan") == 0) {
                backend = NEEDLE_BACKEND_VULKAN;
            } else {
                backend = NEEDLE_BACKEND_CPU;
            }
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--json-schema") == 0 && i + 1 < argc) {
            schema = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    needle_config_t config = {
        .backend = backend,
        .enable_grammar = (schema != NULL),
        .json_schema = schema,
        .temperature = temperature,
        .top_p = 0.9f
    };

    printf("[Needle 2] Opening model '%s' (backend: %s)...\n",
           model_path, (backend == NEEDLE_BACKEND_VULKAN) ? "Vulkan" : "CPU");

    needle_context_t *ctx = needle_open(model_path, config);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to load model binary '%s'\n", model_path);
        return 1;
    }

    size_t prompt_len = strlen(prompt_text);
    uint8_t out_buf[512] = {0};
    needle_output_meta_t meta;

    printf("[Needle 2] Evaluating prompt: '%s'\n", prompt_text);
    clock_t start_time = clock();
    int res = needle_eval(ctx, (const uint8_t *)prompt_text, prompt_len, out_buf, sizeof(out_buf) - 1, &meta);
    clock_t end_time = clock();

    double elapsed_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    double tps = (elapsed_sec > 0.0) ? (meta.generated_tokens / elapsed_sec) : 0.0;

    if (res == 0) {
        printf("[Needle 2] Output: %s\n", (char *)out_buf);
        printf("[Needle 2] Generated tokens: %zu | Time: %.3fs (%.2f tok/s) | Confidence: %.4f\n",
               meta.generated_tokens, elapsed_sec, tps, meta.confidence);
    } else {
        fprintf(stderr, "Error during evaluation.\n");
    }

    needle_close(ctx);
    return 0;
}
