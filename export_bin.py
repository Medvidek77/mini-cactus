#!/usr/bin/env python3
"""
Needle 2 Model Exporter (Python stdlib to binary file format).
Converts model weights and configuration into a contiguous, mmap-friendly binary blob (needle2.bin).

Binary Layout:
  1. Header (128 bytes)
  2. Token Embeddings [vocab_size, dim] (float32)
  3. Engram Hash Tables [engram_vocab_size, engram_dim] (float32)
  4. Per-Layer Weights:
     - q_proj [dim, n_heads * head_dim] (float32)
     - k_proj [dim, n_kv_heads * head_dim] (float32)
     - v_proj [dim, n_kv_heads * head_dim] (float32)
     - o_proj [n_heads * head_dim, dim] (float32)
     - mlp_gate [dim, dim] (float32)
     - mlp_up   [dim, dim] (float32)
     - mlp_down [dim, dim] (float32)
     - norm_attn [dim] (float32)
     - norm_mlp  [dim] (float32)
  5. Final Norm [dim] (float32)
  6. Confidence Head Weights [dim, 1] (float32)
"""

import os
import struct
import random
import argparse

MAGIC = b"NDL2"
VERSION = 1

def generate_float_array(count, scale=0.02, offset=0.0):
    return [random.gauss(0.0, 1.0) * scale + offset for _ in range(count)]

def pack_floats(arr):
    return struct.pack(f"<{len(arr)}f", *arr)

def create_needle2_weights(filename="needle2.bin", dim=256, n_layers=4, n_heads=8, n_kv_heads=2, head_dim=32, vocab_size=256, engram_vocab_size=1024, engram_dim=64):
    max_seq_len = 256

    header_bytes = struct.pack(
        "<4sIIIIIIIIII",
        MAGIC,
        VERSION,
        dim,
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size,
        engram_vocab_size,
        engram_dim,
        max_seq_len
    )
    # Pad header to 128 bytes
    header_bytes = header_bytes.ljust(128, b'\x00')

    random.seed(42)

    with open(filename, "wb") as f:
        f.write(header_bytes)

        # 1. Token Embeddings
        token_emb = generate_float_array(vocab_size * dim, scale=0.02)
        f.write(pack_floats(token_emb))

        # 2. Engram Hash Tables
        engram_tables = generate_float_array(engram_vocab_size * engram_dim, scale=0.02)
        f.write(pack_floats(engram_tables))

        # 3. Layer weights
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        for _ in range(n_layers):
            q_proj = generate_float_array(dim * q_dim, scale=0.02)
            k_proj = generate_float_array(dim * kv_dim, scale=0.02)
            v_proj = generate_float_array(dim * kv_dim, scale=0.02)
            o_proj = generate_float_array(q_dim * dim, scale=0.02)

            mlp_gate = generate_float_array(dim * dim, scale=0.02)
            mlp_up = generate_float_array(dim * dim, scale=0.02)
            mlp_down = generate_float_array(dim * dim, scale=0.02)

            norm_attn = [1.0] * dim
            norm_mlp = [1.0] * dim

            f.write(pack_floats(q_proj))
            f.write(pack_floats(k_proj))
            f.write(pack_floats(v_proj))
            f.write(pack_floats(o_proj))
            f.write(pack_floats(mlp_gate))
            f.write(pack_floats(mlp_up))
            f.write(pack_floats(mlp_down))
            f.write(pack_floats(norm_attn))
            f.write(pack_floats(norm_mlp))

        # 4. Final norm
        final_norm = [1.0] * dim
        f.write(pack_floats(final_norm))

        # 5. Confidence Head Weights
        conf_head = generate_float_array(dim * 1, scale=0.02)
        f.write(pack_floats(conf_head))

    print(f"[Export] Successfully generated binary model '{filename}' ({os.path.getsize(filename)} bytes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Needle 2 model to raw mmap binary file format.")
    parser.add_argument("--output", type=str, default="needle2.bin", help="Output binary file path")
    args = parser.parse_args()
    create_needle2_weights(filename=args.output)
