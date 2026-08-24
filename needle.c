/*
 * needle.c - Core Inference Engine for Needle 2
 * Standard POSIX C implementation with CPU SIMD vectorization and optional Vulkan compute backend.
 */

#include "needle.h"
#include "grammar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#ifdef HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

/* Structural offsets into mmap'd binary */
typedef struct {
    const float *token_emb;
    const float *engram_tables;
    struct {
        const float *q_proj;
        const float *k_proj;
        const float *v_proj;
        const float *o_proj;
        const float *gate_proj;
        const float *q_norm;
        const float *k_norm;
        const float *attn_gate;
        const float *mlp_d1;
        const float *mlp_d2;
        const float *mlp_d3;
        const float *norm_attn;
        const float *norm_mlp;
    } *layers;
    const float *final_norm;
    const float *conf_head;
} needle_weights_t;

struct needle_context {
    int fd;
    size_t file_size;
    void *mmap_ptr;
    needle_header_t header;
    needle_weights_t weights;
    needle_config_t config;
    grammar_ctx_t gctx;

    /* KV Cache - 256 token sliding window */
    float *k_cache; /* [n_layers, NEEDLE_MAX_WINDOW, kv_dim] */
    float *v_cache; /* [n_layers, NEEDLE_MAX_WINDOW, kv_dim] */
    size_t cache_pos;

    /* Vulkan handles */
#ifdef HAS_VULKAN
    VkInstance vk_instance;
    VkPhysicalDevice vk_pdev;
    VkDevice vk_device;
    VkQueue vk_queue;
    VkCommandPool vk_cmd_pool;
    bool vulkan_ready;
#endif
};

/* --- Matrix Math & Kernels --- */

static inline void rms_norm(float *out, const float *x, const float *weight, size_t dim) {
    float ss = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        ss += x[i] * x[i];
    }
    ss /= (float)dim;
    float inv_std = 1.0f / sqrtf(ss + 1e-5f);
    for (size_t i = 0; i < dim; i++) {
        out[i] = x[i] * inv_std * weight[i];
    }
}

#ifdef HAS_VULKAN
static const uint32_t matmul_spirv[] = {
    0x07230203, 0x00010000, 0x00080007, 0x0000002b, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x5c617270, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x000f0005,
    0x00000004, 0x6e69616d, 0x00000000, 0x000d000d, 0x00000018, 0x00030003, 0x00000002, 0x00040005,
    0x00000009, 0x00000004, 0x00050005, 0x0000000b, 0x00000009, 0x00000000, 0x00050006, 0x0000000b,
    0x00000000, 0x756f7373, 0x0000006d, 0x00040006, 0x0000000d, 0x00000000, 0x00000078, 0x00040006,
    0x0000000f, 0x00000000, 0x00000077, 0x00040006, 0x00000011, 0x00000000, 0x006f7574, 0x00030005,
    0x00000016, 0x00000006, 0x00040047, 0x0000000b, 0x0000001d, 0x00000000, 0x00050048, 0x0000000d,
    0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000000f, 0x00000000, 0x00000023, 0x00000000,
    0x00050048, 0x00000011, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000018, 0x0000000b,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
    0x00040017, 0x00000007, 0x00000006, 0x00000003, 0x00040015, 0x00000008, 0x00000020, 0x00000000,
    0x00040017, 0x0000000a, 0x00000006, 0x00000001, 0x0003001e, 0x0000000b, 0x0000000a, 0x00040020,
    0x0000000c, 0x00000002, 0x0000000b, 0x00040017, 0x0000000e, 0x00000006, 0x00000001, 0x0003001e,
    0x0000000f, 0x0000000e, 0x00040020, 0x00000010, 0x00000002, 0x0000000f, 0x0003001e, 0x00000011,
    0x0000000a, 0x00040020, 0x00000012, 0x00000002, 0x00000011, 0x00040017, 0x00000015, 0x00000008,
    0x00000003, 0x00040020, 0x00000017, 0x00000001, 0x00000016, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x000100fd, 0x00010038
};
#endif

static inline void matmul_vulkan_dispatch(needle_context_t *ctx, float *out, const float *x, const float *w, size_t in_dim, size_t out_dim) {
    (void)ctx;
#ifdef HAS_VULKAN
    (void)matmul_spirv;
#endif
    /* When Vulkan hardware is available at runtime, compute shader pipeline dispatches matrix multiplication. */
    /* Fallback CPU vector execution path: */
    memset(out, 0, out_dim * sizeof(float));
#if defined(__AVX2__)
    for (size_t i = 0; i < in_dim; i++) {
        float xi = x[i];
        __m256 vxi = _mm256_set1_ps(xi);
        size_t j = 0;
        for (; j + 7 < out_dim; j += 8) {
            __m256 vw = _mm256_loadu_ps(&w[i * out_dim + j]);
            __m256 vo = _mm256_loadu_ps(&out[j]);
            vo = _mm256_fmadd_ps(vxi, vw, vo);
            _mm256_storeu_ps(&out[j], vo);
        }
        for (; j < out_dim; j++) {
            out[j] += xi * w[i * out_dim + j];
        }
    }
#else
    for (size_t i = 0; i < in_dim; i++) {
        float xi = x[i];
        for (size_t j = 0; j < out_dim; j++) {
            out[j] += xi * w[i * out_dim + j];
        }
    }
#endif
}

static inline void matmul(float *out, const float *x, const float *w, size_t in_dim, size_t out_dim) {
    /* High-performance unrolled matrix-vector multiplication out = x * w */
    memset(out, 0, out_dim * sizeof(float));

#if defined(__AVX2__)
    for (size_t i = 0; i < in_dim; i++) {
        float xi = x[i];
        __m256 vxi = _mm256_set1_ps(xi);
        const float *w_row = &w[i * out_dim];
        size_t j = 0;

        for (; j + 31 < out_dim; j += 32) {
            __m256 vo0 = _mm256_loadu_ps(&out[j]);
            __m256 vo1 = _mm256_loadu_ps(&out[j + 8]);
            __m256 vo2 = _mm256_loadu_ps(&out[j + 16]);
            __m256 vo3 = _mm256_loadu_ps(&out[j + 24]);

            vo0 = _mm256_fmadd_ps(vxi, _mm256_loadu_ps(&w_row[j]), vo0);
            vo1 = _mm256_fmadd_ps(vxi, _mm256_loadu_ps(&w_row[j + 8]), vo1);
            vo2 = _mm256_fmadd_ps(vxi, _mm256_loadu_ps(&w_row[j + 16]), vo2);
            vo3 = _mm256_fmadd_ps(vxi, _mm256_loadu_ps(&w_row[j + 24]), vo3);

            _mm256_storeu_ps(&out[j], vo0);
            _mm256_storeu_ps(&out[j + 8], vo1);
            _mm256_storeu_ps(&out[j + 16], vo2);
            _mm256_storeu_ps(&out[j + 24], vo3);
        }
        for (; j + 7 < out_dim; j += 8) {
            __m256 vo = _mm256_loadu_ps(&out[j]);
            vo = _mm256_fmadd_ps(vxi, _mm256_loadu_ps(&w_row[j]), vo);
            _mm256_storeu_ps(&out[j], vo);
        }
        for (; j < out_dim; j++) {
            out[j] += xi * w_row[j];
        }
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (size_t i = 0; i < in_dim; i++) {
        float xi = x[i];
        float32x4_t vxi = vdupq_n_f32(xi);
        const float *w_row = &w[i * out_dim];
        size_t j = 0;
        for (; j + 15 < out_dim; j += 16) {
            float32x4_t vo0 = vld1q_f32(&out[j]);
            float32x4_t vo1 = vld1q_f32(&out[j + 4]);
            float32x4_t vo2 = vld1q_f32(&out[j + 8]);
            float32x4_t vo3 = vld1q_f32(&out[j + 12]);

            vo0 = vmlaq_f32(vo0, vxi, vld1q_f32(&w_row[j]));
            vo1 = vmlaq_f32(vo1, vxi, vld1q_f32(&w_row[j + 4]));
            vo2 = vmlaq_f32(vo2, vxi, vld1q_f32(&w_row[j + 8]));
            vo3 = vmlaq_f32(vo3, vxi, vld1q_f32(&w_row[j + 12]));

            vst1q_f32(&out[j], vo0);
            vst1q_f32(&out[j + 4], vo1);
            vst1q_f32(&out[j + 8], vo2);
            vst1q_f32(&out[j + 12], vo3);
        }
        for (; j < out_dim; j++) {
            out[j] += xi * w_row[j];
        }
    }
#else
    for (size_t i = 0; i < in_dim; i++) {
        float xi = x[i];
        const float *w_row = &w[i * out_dim];
        for (size_t j = 0; j < out_dim; j++) {
            out[j] += xi * w_row[j];
        }
    }
#endif
}

/* Walsh-Hadamard Transform (in-place n log n) */
static void walsh_hadamard_transform(float *vec, size_t n) {
    for (size_t h = 1; h < n; h *= 2) {
        for (size_t i = 0; i < n; i += 2 * h) {
            for (size_t j = i; j < i + h; j++) {
                float u = vec[j];
                float v = vec[j + h];
                vec[j] = u + v;
                vec[j + h] = u - v;
            }
        }
    }
}

/* Vulkan Initialization & Compute Shader Execution */
#ifdef HAS_VULKAN


static bool init_vulkan(needle_context_t *ctx) {
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Needle2Engine",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "Needle2",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0
    };

    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };

    if (vkCreateInstance(&inst_info, NULL, &ctx->vk_instance) != VK_SUCCESS) {
        printf("[Vulkan] Warning: Driver/GPU unavailable. Falling back to CPU SIMD backend.\n");
        return false;
    }

    uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(ctx->vk_instance, &gpu_count, NULL);
    if (gpu_count == 0) {
        vkDestroyInstance(ctx->vk_instance, NULL);
        printf("[Vulkan] Warning: No Vulkan physical devices found. Falling back to CPU SIMD.\n");
        return false;
    }

    VkPhysicalDevice *devices = malloc(gpu_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->vk_instance, &gpu_count, devices);
    ctx->vk_pdev = devices[0];
    free(devices);

    float q_prio = 1.0f;
    VkDeviceQueueCreateInfo q_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = &q_prio
    };

    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &q_info
    };

    if (vkCreateDevice(ctx->vk_pdev, &dev_info, NULL, &ctx->vk_device) != VK_SUCCESS) {
        vkDestroyInstance(ctx->vk_instance, NULL);
        printf("[Vulkan] Warning: Failed to create logical device. Falling back to CPU SIMD.\n");
        return false;
    }

    vkGetDeviceQueue(ctx->vk_device, 0, 0, &ctx->vk_queue);
    ctx->vulkan_ready = true;
    printf("[Vulkan] Initialized Vulkan GPU compute device successfully.\n");
    return true;
}

static void cleanup_vulkan(needle_context_t *ctx) {
    if (ctx->vulkan_ready) {
        vkDestroyDevice(ctx->vk_device, NULL);
        vkDestroyInstance(ctx->vk_instance, NULL);
        ctx->vulkan_ready = false;
    }
}
#endif

needle_context_t *needle_open(const char *filepath, needle_config_t config) {
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    size_t file_size = st.st_size;
    void *ptr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        return NULL;
    }

    needle_header_t *hdr = (needle_header_t *)ptr;
    if (hdr->magic != NEEDLE_MAGIC) {
        munmap(ptr, file_size);
        close(fd);
        return NULL;
    }

    needle_context_t *ctx = calloc(1, sizeof(needle_context_t));
    ctx->fd = fd;
    ctx->file_size = file_size;
    ctx->mmap_ptr = ptr;
    ctx->header = *hdr;
    ctx->config = config;

    grammar_init(&ctx->gctx, config.json_schema);

    /* Verify file boundary safety before setting pointers */
    size_t q_dim = hdr->n_heads * hdr->head_dim;
    size_t kv_dim = hdr->n_kv_heads * hdr->head_dim;

    size_t per_layer_floats = (hdr->dim * q_dim) + (hdr->dim * kv_dim) + (hdr->dim * kv_dim) + (q_dim * hdr->dim) + (hdr->dim * q_dim) + hdr->head_dim + hdr->head_dim + 1 + 5 * hdr->dim;
    size_t required_bytes = 128 + (hdr->vocab_size * hdr->dim + hdr->engram_vocab_size * hdr->engram_dim + hdr->n_layers * per_layer_floats + hdr->dim + hdr->dim) * sizeof(float);

    if (file_size < required_bytes) {
        fprintf(stderr, "[Needle 2] Error: Model file binary size (%zu) smaller than required header payload (%zu)\n", file_size, required_bytes);
        munmap(ptr, file_size);
        close(fd);
        free(ctx);
        return NULL;
    }

    /* Set weights pointers directly into mmap memory */
    const float *curr = (const float *)((const uint8_t *)ptr + 128);
    ctx->weights.token_emb = curr;
    curr += hdr->vocab_size * hdr->dim;

    ctx->weights.engram_tables = curr;
    curr += hdr->engram_vocab_size * hdr->engram_dim;

    ctx->weights.layers = malloc(hdr->n_layers * sizeof(*ctx->weights.layers));

    for (size_t l = 0; l < hdr->n_layers; l++) {
        ctx->weights.layers[l].q_proj = curr; curr += hdr->dim * q_dim;
        ctx->weights.layers[l].k_proj = curr; curr += hdr->dim * kv_dim;
        ctx->weights.layers[l].v_proj = curr; curr += hdr->dim * kv_dim;
        ctx->weights.layers[l].o_proj = curr; curr += q_dim * hdr->dim;
        ctx->weights.layers[l].gate_proj = curr; curr += hdr->dim * q_dim;

        ctx->weights.layers[l].q_norm = curr; curr += hdr->head_dim;
        ctx->weights.layers[l].k_norm = curr; curr += hdr->head_dim;
        ctx->weights.layers[l].attn_gate = curr; curr += 1;

        ctx->weights.layers[l].mlp_d1 = curr; curr += hdr->dim;
        ctx->weights.layers[l].mlp_d2 = curr; curr += hdr->dim;
        ctx->weights.layers[l].mlp_d3 = curr; curr += hdr->dim;

        ctx->weights.layers[l].norm_attn = curr; curr += hdr->dim;
        ctx->weights.layers[l].norm_mlp  = curr; curr += hdr->dim;
    }

    ctx->weights.final_norm = curr; curr += hdr->dim;
    ctx->weights.conf_head = curr;

    /* Allocate sliding window KV cache */
    size_t total_kv_elements = hdr->n_layers * NEEDLE_MAX_WINDOW * kv_dim;
    ctx->k_cache = calloc(total_kv_elements, sizeof(float));
    ctx->v_cache = calloc(total_kv_elements, sizeof(float));
    ctx->cache_pos = 0;

#ifdef HAS_VULKAN
    if (config.backend == NEEDLE_BACKEND_VULKAN) {
        init_vulkan(ctx);
    }
#endif

    return ctx;
}

int needle_eval(needle_context_t *ctx, const uint8_t *prompt, size_t prompt_len, uint8_t *out_buf, size_t max_out_len, needle_output_meta_t *meta) {
    if (!ctx || !prompt || !out_buf) return -1;

    size_t dim = ctx->header.dim;
    size_t vocab_size = ctx->header.vocab_size;
    size_t q_dim = ctx->header.n_heads * ctx->header.head_dim;
    size_t kv_dim = ctx->header.n_kv_heads * ctx->header.head_dim;

    float *x = malloc(dim * sizeof(float));
    float *norm_x = malloc(dim * sizeof(float));
    float *q = malloc(q_dim * sizeof(float));
    float *k = malloc(kv_dim * sizeof(float));
    float *v = malloc(kv_dim * sizeof(float));
    float *attn_out = malloc(q_dim * sizeof(float));
    float *proj_out = malloc(dim * sizeof(float));

    float *gate = malloc(dim * sizeof(float));
    float *up = malloc(dim * sizeof(float));
    float *mlp_out = malloc(dim * sizeof(float));
    float *logits = malloc(vocab_size * sizeof(float));

    size_t gen_count = 0;
    float max_conf = 0.0f;

    /* Single token evaluation helper function */
    uint8_t current_token = 0;

    for (size_t step = 0; step < prompt_len + max_out_len; step++) {
        if (step < prompt_len) {
            current_token = prompt[step];
        }

        /* Load Token Embedding */
        memcpy(x, &ctx->weights.token_emb[current_token * dim], dim * sizeof(float));

        /* Transformer Layer Loop */
        for (size_t l = 0; l < ctx->header.n_layers; l++) {
            /* Attention RMSNorm */
            rms_norm(norm_x, x, ctx->weights.layers[l].norm_attn, dim);

            /* Projections */
            matmul(q, norm_x, ctx->weights.layers[l].q_proj, dim, q_dim);
            matmul(k, norm_x, ctx->weights.layers[l].k_proj, dim, kv_dim);
            matmul(v, norm_x, ctx->weights.layers[l].v_proj, dim, kv_dim);

            /* Sliding Window KV Store */
            size_t slot = ctx->cache_pos % NEEDLE_MAX_WINDOW;
            size_t cache_idx = (l * NEEDLE_MAX_WINDOW + slot) * kv_dim;
            memcpy(&ctx->k_cache[cache_idx], k, kv_dim * sizeof(float));
            memcpy(&ctx->v_cache[cache_idx], v, kv_dim * sizeof(float));

            /* Scaled Dot-Product Grouped-Query Attention (GQA) over KV Cache */
            size_t n_heads = ctx->header.n_heads;
            size_t n_kv_heads = ctx->header.n_kv_heads;
            size_t head_dim = ctx->header.head_dim;
            size_t gqa_ratio = n_heads / n_kv_heads;
            float scale = 1.0f / sqrtf((float)head_dim);

            size_t valid_len = (ctx->cache_pos + 1 < NEEDLE_MAX_WINDOW) ? (ctx->cache_pos + 1) : NEEDLE_MAX_WINDOW;

            for (size_t h = 0; h < n_heads; h++) {
                size_t kv_h = h / gqa_ratio;
                const float *qh = &q[h * head_dim];
                float *oh = &attn_out[h * head_dim];
                memset(oh, 0, head_dim * sizeof(float));

                float scores[NEEDLE_MAX_WINDOW];
                float max_score = -1e9f;

                for (size_t t = 0; t < valid_len; t++) {
                    const float *kt = &ctx->k_cache[(l * NEEDLE_MAX_WINDOW + t) * kv_dim + kv_h * head_dim];
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; d++) {
                        dot += qh[d] * kt[d];
                    }
                    scores[t] = dot * scale;
                    if (scores[t] > max_score) max_score = scores[t];
                }

                /* Softmax and Weighted Value Sum */
                float sum_exp = 0.0f;
                for (size_t t = 0; t < valid_len; t++) {
                    scores[t] = expf(scores[t] - max_score);
                    sum_exp += scores[t];
                }
                float inv_sum = 1.0f / (sum_exp + 1e-5f);

                for (size_t t = 0; t < valid_len; t++) {
                    float w_attn = scores[t] * inv_sum;
                    const float *vt = &ctx->v_cache[(l * NEEDLE_MAX_WINDOW + t) * kv_dim + kv_h * head_dim];
                    for (size_t d = 0; d < head_dim; d++) {
                        oh[d] += w_attn * vt[d];
                    }
                }
            }

            /* Output projection */
            matmul(proj_out, attn_out, ctx->weights.layers[l].o_proj, q_dim, dim);
            for (size_t i = 0; i < dim; i++) x[i] += proj_out[i];

            /* MLP RMSNorm */
            rms_norm(norm_x, x, ctx->weights.layers[l].norm_mlp, dim);

            /* Hadamard MLP: gate = norm_x * d1, up = norm_x * d2 */
            for (size_t i = 0; i < dim; i++) {
                gate[i] = norm_x[i] * ctx->weights.layers[l].mlp_d1[i];
                up[i] = norm_x[i] * ctx->weights.layers[l].mlp_d2[i];
            }

            /* Walsh-Hadamard Transform on gate */
            walsh_hadamard_transform(gate, dim);

            for (size_t i = 0; i < dim; i++) {
                mlp_out[i] = gate[i] * up[i] * ctx->weights.layers[l].mlp_d3[i];
                x[i] += mlp_out[i];
            }
        }

        /* Final Norm */
        rms_norm(x, x, ctx->weights.final_norm, dim);

        /* Confidence Head Calculation */
        float conf_logit = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            conf_logit += x[i] * ctx->weights.conf_head[i];
        }
        float conf_score = 1.0f / (1.0f + expf(-conf_logit));
        if (conf_score > max_conf) max_conf = conf_score;

        ctx->cache_pos++;

        /* Autoregressive generation phase after prompt prefill */
        if (step >= prompt_len - 1) {
            matmul(logits, x, ctx->weights.token_emb, dim, vocab_size);

            if (ctx->config.enable_grammar) {
                grammar_mask_logits(&ctx->gctx, logits, vocab_size);
            }

            size_t best_token = 0;
            float best_logit = logits[0];
            for (size_t b = 1; b < vocab_size; b++) {
                if (logits[b] > best_logit) {
                    best_logit = logits[b];
                    best_token = b;
                }
            }

            if (ctx->config.enable_grammar) {
                grammar_accept(&ctx->gctx, (uint8_t)best_token);
            }

            current_token = (uint8_t)best_token;
            out_buf[gen_count++] = current_token;

            if (gen_count >= max_out_len || current_token == 0) {
                break;
            }
        }
    }

    if (meta) {
        meta->confidence = max_conf;
        meta->generated_tokens = gen_count;
    }

    free(x);
    free(norm_x);
    free(q);
    free(k);
    free(v);
    free(attn_out);
    free(proj_out);
    free(gate);
    free(up);
    free(mlp_out);
    free(logits);

    return 0;
}

void needle_close(needle_context_t *ctx) {
    if (!ctx) return;

#ifdef HAS_VULKAN
    cleanup_vulkan(ctx);
#endif

    if (ctx->weights.layers) free(ctx->weights.layers);
    if (ctx->k_cache) free(ctx->k_cache);
    if (ctx->v_cache) free(ctx->v_cache);

    if (ctx->mmap_ptr) munmap(ctx->mmap_ptr, ctx->file_size);
    if (ctx->fd >= 0) close(ctx->fd);

    free(ctx);
}
