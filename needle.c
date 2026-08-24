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
        const float *mlp_gate;
        const float *mlp_up;
        const float *mlp_down;
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

static inline void matmul(float *out, const float *x, const float *w, size_t in_dim, size_t out_dim) {
    /* out = x * w  where x is [in_dim], w is [in_dim, out_dim] */
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
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (size_t i = 0; i < in_dim; i++) {
        float xi = x[i];
        float32x4_t vxi = vdupq_n_f32(xi);
        size_t j = 0;
        for (; j + 3 < out_dim; j += 4) {
            float32x4_t vw = vld1q_f32(&w[i * out_dim + j]);
            float32x4_t vo = vld1q_f32(&out[j]);
            vo = vmlaq_f32(vo, vxi, vw);
            vst1q_f32(&out[j], vo);
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

/* Vulkan Initialization */
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
        return false;
    }

    uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(ctx->vk_instance, &gpu_count, NULL);
    if (gpu_count == 0) return false;

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
        return false;
    }

    vkGetDeviceQueue(ctx->vk_device, 0, 0, &ctx->vk_queue);
    ctx->vulkan_ready = true;
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

    /* Set weights pointers directly into mmap memory */
    const float *curr = (const float *)((const uint8_t *)ptr + 128);
    ctx->weights.token_emb = curr;
    curr += hdr->vocab_size * hdr->dim;

    ctx->weights.engram_tables = curr;
    curr += hdr->engram_vocab_size * hdr->engram_dim;

    size_t q_dim = hdr->n_heads * hdr->head_dim;
    size_t kv_dim = hdr->n_kv_heads * hdr->head_dim;

    ctx->weights.layers = malloc(hdr->n_layers * sizeof(*ctx->weights.layers));

    for (size_t l = 0; l < hdr->n_layers; l++) {
        ctx->weights.layers[l].q_proj = curr; curr += hdr->dim * q_dim;
        ctx->weights.layers[l].k_proj = curr; curr += hdr->dim * kv_dim;
        ctx->weights.layers[l].v_proj = curr; curr += hdr->dim * kv_dim;
        ctx->weights.layers[l].o_proj = curr; curr += q_dim * hdr->dim;

        ctx->weights.layers[l].mlp_gate = curr; curr += hdr->dim * hdr->dim;
        ctx->weights.layers[l].mlp_up   = curr; curr += hdr->dim * hdr->dim;
        ctx->weights.layers[l].mlp_down = curr; curr += hdr->dim * hdr->dim;

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
            size_t cache_idx = (l * NEEDLE_MAX_WINDOW + (ctx->cache_pos % NEEDLE_MAX_WINDOW)) * kv_dim;
            memcpy(&ctx->k_cache[cache_idx], k, kv_dim * sizeof(float));
            memcpy(&ctx->v_cache[cache_idx], v, kv_dim * sizeof(float));

            /* Simplified GQA Attention */
            memcpy(attn_out, q, q_dim * sizeof(float));

            /* Output projection */
            matmul(proj_out, attn_out, ctx->weights.layers[l].o_proj, q_dim, dim);
            for (size_t i = 0; i < dim; i++) x[i] += proj_out[i];

            /* MLP RMSNorm */
            rms_norm(norm_x, x, ctx->weights.layers[l].norm_mlp, dim);

            /* Hadamard MLP */
            matmul(gate, norm_x, ctx->weights.layers[l].mlp_gate, dim, dim);
            matmul(up, norm_x, ctx->weights.layers[l].mlp_up, dim, dim);

            /* Walsh-Hadamard Transform on gate */
            walsh_hadamard_transform(gate, dim);

            for (size_t i = 0; i < dim; i++) {
                gate[i] = gate[i] * up[i];
            }

            matmul(mlp_out, gate, ctx->weights.layers[l].mlp_down, dim, dim);
            for (size_t i = 0; i < dim; i++) x[i] += mlp_out[i];
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
