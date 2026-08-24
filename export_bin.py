#!/usr/bin/env python3
"""
Needle 2 Model Exporter
Converts model weights (.pkl / .cact / JAX checkpoint) into contiguous needle2.bin.
Pure Python implementation with optional numpy/pickle loading for real checkpoints.
"""

import os
import struct
import random
import argparse

MAGIC = b"NDL2"
VERSION = 1

def generate_floats(count, scale=0.02, offset=0.0):
    return [random.gauss(0.0, 1.0) * scale + offset for _ in range(count)]

def pack_floats(arr):
    return struct.pack(f"<{len(arr)}f", *arr)

def generate_dummy_weights(output_bin="needle2.bin", dim=512, n_layers=27, n_heads=8, n_kv_heads=4, head_dim=64, vocab_size=8192, engram_vocab_size=8192, engram_dim=128):
    max_seq_len = 2048
    header = struct.pack(
        "<4sIIIIIIIIII",
        MAGIC, VERSION, dim, n_layers, n_heads, n_kv_heads,
        head_dim, vocab_size, engram_vocab_size, engram_dim, max_seq_len
    ).ljust(128, b'\x00')

    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    random.seed(42)

    with open(output_bin, "wb") as f:
        f.write(header)
        f.write(pack_floats(generate_floats(vocab_size * dim)))
        f.write(pack_floats(generate_floats(engram_vocab_size * engram_dim)))

        for _ in range(n_layers):
            f.write(pack_floats(generate_floats(dim * q_dim)))
            f.write(pack_floats(generate_floats(dim * kv_dim)))
            f.write(pack_floats(generate_floats(dim * kv_dim)))
            f.write(pack_floats(generate_floats(q_dim * dim)))
            f.write(pack_floats(generate_floats(dim * q_dim)))

            f.write(pack_floats([1.0] * head_dim))
            f.write(pack_floats([1.0] * head_dim))
            f.write(pack_floats([1.0]))

            f.write(pack_floats([1.0] * dim))
            f.write(pack_floats([1.0] * dim))
            f.write(pack_floats([1.0] * dim))

            f.write(pack_floats([1.0] * dim))
            f.write(pack_floats([1.0] * dim))

        f.write(pack_floats([1.0] * dim))
        f.write(pack_floats(generate_floats(dim)))

    print(f"[Export] Dummy binary model '{output_bin}' exported ({os.path.getsize(output_bin)} bytes)")

def export_checkpoint(pkl_path, output_bin="needle2.bin"):
    if not os.path.exists(pkl_path):
        print(f"[*] Checkpoint '{pkl_path}' not found, generating dummy binary model...")
        generate_dummy_weights(output_bin=output_bin)
        return

    try:
        import pickle
        import numpy as np
    except ImportError:
        print("[!] numpy/pickle module missing. Falling back to dummy generator.")
        generate_dummy_weights(output_bin=output_bin)
        return

    print(f"[*] Loading model checkpoint from '{pkl_path}'...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    cfg = data.get("config", {})
    params = data.get("params", {})
    stack = params.get("stack", {})
    block = stack.get("layers", {}).get("block", {})
    self_attn = block.get("self_attn", {})
    mlp = block.get("hadamard_mlp", {})

    dim = cfg.get("d_model", 512)
    n_layers = cfg.get("num_layers", 27)
    n_heads = cfg.get("num_heads", 8)
    n_kv_heads = cfg.get("num_kv_heads", 4)
    head_dim = dim // n_heads
    vocab_size = cfg.get("vocab_size", 8192)
    engram_vocab_size = cfg.get("engram_slots", 8192)
    engram_dim = 128
    max_seq_len = cfg.get("max_seq_len", 2048)

    header = struct.pack(
        "<4sIIIIIIIIII",
        MAGIC, VERSION, dim, n_layers, n_heads, n_kv_heads,
        head_dim, vocab_size, engram_vocab_size, engram_dim, max_seq_len
    ).ljust(128, b'\x00')

    def pack(arr):
        return np.ascontiguousarray(arr, dtype=np.float32).tobytes()

    with open(output_bin, "wb") as f:
        f.write(header)

        # 1. Token Embeddings [8192, 512]
        emb = params["embedding"]["embedding"]
        f.write(pack(emb))

        # 2. Engram Hash Tables [8192, 128]
        engram = params.get("engrams_0", {}).get("embedding", np.zeros((4, engram_vocab_size, engram_dim)))[0]
        f.write(pack(engram))

        # 3. Layer weights
        for i in range(n_layers):
            f.write(pack(self_attn["q_proj"]["kernel"][i]))
            f.write(pack(self_attn["k_proj"]["kernel"][i]))
            f.write(pack(self_attn["v_proj"]["kernel"][i]))
            f.write(pack(self_attn["out_proj"]["kernel"][i]))

            if "gate_proj" in self_attn:
                f.write(pack(self_attn["gate_proj"]["kernel"][i]))
            else:
                f.write(pack(np.ones((dim, n_heads * head_dim), dtype=np.float32)))

            if "q_norm" in self_attn:
                f.write(pack(self_attn["q_norm"]["scale"][i]))
            else:
                f.write(pack(np.ones(head_dim, dtype=np.float32)))

            if "k_norm" in self_attn:
                f.write(pack(self_attn["k_norm"]["scale"][i]))
            else:
                f.write(pack(np.ones(head_dim, dtype=np.float32)))

            if "attn_gate" in block:
                f.write(pack(block["attn_gate"][i]))
            else:
                f.write(pack(np.array([1.0], dtype=np.float32)))

            f.write(pack(mlp["d1"][i]))
            f.write(pack(mlp["d2"][i]))
            f.write(pack(mlp["d3"][i]))

            if "pre_hada_norm" in block:
                f.write(pack(block["pre_hada_norm"]["scale"][i]))
            else:
                f.write(pack(np.ones(dim, dtype=np.float32)))

            if "post_attn_norm" in block:
                f.write(pack(block["post_attn_norm"]["scale"][i]))
            else:
                f.write(pack(np.ones(dim, dtype=np.float32)))

        # 4. Final Norm
        f.write(pack(stack["final_norm"]["scale"]))

        # 5. Confidence Head
        conf_kernel = params.get("confidence_head", {}).get("proj", {}).get("kernel", np.zeros((dim, 1)))
        f.write(pack(conf_kernel[:dim, :1]))

    size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print(f"[Export] '{output_bin}' exported successfully ({size_mb:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="needle2.pkl")
    parser.add_argument("--output", default="needle2.bin")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)
