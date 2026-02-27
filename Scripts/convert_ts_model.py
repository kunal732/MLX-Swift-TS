#!/usr/bin/env python3
"""
Convert time series foundation models to MLX safetensors format.

Supports: Toto, Chronos (v1 & v2), TimesFM (v2.0 & v2.5), Lag-Llama,
          FlowState, Kairos, TiRex.

Usage:
    python convert_ts_model.py --hf-path Datadog/Toto-Open-Base-1.0 --mlx-path ./converted/toto
    python convert_ts_model.py --hf-path amazon/chronos-t5-base --mlx-path ./converted/chronos
    python convert_ts_model.py --hf-path autogluon/chronos-2-synth --mlx-path ./converted/chronos2
    python convert_ts_model.py --hf-path google/timesfm-2.5-200m-pytorch --mlx-path ./converted/timesfm
    python convert_ts_model.py --hf-path time-series-foundation-models/Lag-Llama --mlx-path ./converted/lag-llama
    python convert_ts_model.py --hf-path ibm-granite/granite-timeseries-flowstate-r1 --mlx-path ./converted/flowstate
    python convert_ts_model.py --hf-path mldi-lab/Kairos_50m --mlx-path ./converted/kairos
    python convert_ts_model.py --hf-path NX-AI/TiRex --mlx-path ./converted/tirex
    python convert_ts_model.py --hf-path Datadog/Toto-Open-Base-1.0 --mlx-path ./converted/toto-4bit -q --q-bits 4
    python convert_ts_model.py --hf-path Datadog/Toto-Open-Base-1.0 -q --upload-repo mlx-community/Toto-Open-Base-1.0-4bit
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

MODEL_TYPE_PATTERNS = {
    "toto": ["toto", "Toto"],
    "chronos_v2": ["chronos-2", "chronos_2"],
    "chronos": ["chronos"],
    "timesfm": ["timesfm", "TimesFM"],
    "lag_llama": ["lag-llama", "Lag-Llama", "lag_llama"],
    "flowstate": ["flowstate", "FlowState"],
    "kairos": ["kairos", "Kairos"],
    "tirex": ["tirex", "TiRex"],
}


def detect_model_type(model_id: str) -> str:
    """Detect model type from the HuggingFace model ID."""
    model_lower = model_id.lower()
    for model_type, patterns in MODEL_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in model_lower:
                return model_type
    raise ValueError(
        f"Cannot detect model type from '{model_id}'. "
        f"Supported patterns: {MODEL_TYPE_PATTERNS}"
    )


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------


def load_safetensors_weights(directory: Path) -> dict[str, mx.array]:
    """Load weights from all safetensors files in a directory."""
    from safetensors import safe_open

    weights = {}
    for f in sorted(directory.glob("*.safetensors")):
        with safe_open(str(f), framework="numpy") as sf:
            for key in sf.keys():
                weights[key] = mx.array(sf.get_tensor(key))
    return weights


def load_pytorch_weights(path: Path) -> dict[str, mx.array]:
    """Load weights from a PyTorch checkpoint (.ckpt or .bin)."""
    import torch

    if path.suffix == ".ckpt":
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)

    weights = {}
    for key, tensor in state_dict.items():
        weights[key] = mx.array(tensor.float().numpy())
    return weights


def load_hf_config(directory: Path) -> dict:
    """Load config.json from a HuggingFace model directory."""
    config_path = directory / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Toto conversion
# ---------------------------------------------------------------------------


def convert_toto(model_dir: Path) -> tuple[dict[str, mx.array], dict]:
    """Convert Toto model to MLX format."""
    print("Converting Toto model...")
    weights = load_safetensors_weights(model_dir)
    hf_config = load_hf_config(model_dir)

    remapped = {}
    for key, value in weights.items():
        if "rotary_emb.freqs" in key or "rotary_emb.scale" in key:
            continue
        new_key = key
        if new_key.endswith(".norm1.scale") or new_key.endswith(".norm2.scale"):
            new_key = new_key[: -len("scale")] + "weight"
        new_key = new_key.replace(".mlp.0.w12.", ".mlp.gate_up.")
        new_key = new_key.replace(".mlp.0.w3.", ".mlp.down.")
        remapped[new_key] = value

    config = {
        "model_type": "toto",
        "ts_model_class": "TotoModel",
        "hidden_size": hf_config.get("embed_dim", 768),
        "num_layers": hf_config.get("num_layers", 12),
        "num_heads": hf_config.get("num_heads", 12),
        "input_format": "patches",
        "output_format": "mixture_student_t",
        "context_length": 4096,
        "prediction_length": 64,
        "patch_size": hf_config.get("patch_size", 64),
        "stride": hf_config.get("stride", 64),
        "embed_dim": hf_config.get("embed_dim", 768),
        "mlp_hidden_dim": hf_config.get("mlp_hidden_dim", 3072),
        "dropout": hf_config.get("dropout", 0.1),
        "spacewise_every_n_layers": hf_config.get("spacewise_every_n_layers", 12),
        "spacewise_first": hf_config.get("spacewise_first", False),
        "output_distribution_kwargs": hf_config.get(
            "output_distribution_kwargs", {"k_components": 24}
        ),
        "use_memory_efficient_attention": hf_config.get(
            "use_memory_efficient_attention", True
        ),
        "stabilize_with_global": hf_config.get("stabilize_with_global", True),
        "scale_factor_exponent": hf_config.get("scale_factor_exponent", 10.0),
    }

    print(f"  Loaded {len(remapped)} weight tensors")
    return remapped, config


# ---------------------------------------------------------------------------
# Chronos v1 conversion (encoder-decoder T5)
# ---------------------------------------------------------------------------


def convert_chronos(model_dir: Path) -> tuple[dict[str, mx.array], dict]:
    """Convert Chronos v1 (T5-based encoder-decoder) to MLX format."""
    print("Converting Chronos v1 model...")
    weights = load_safetensors_weights(model_dir)
    hf_config = load_hf_config(model_dir)

    remapped = {}
    for key, value in weights.items():
        if key == "lm_head.weight" and "shared.weight" in weights:
            continue
        remapped[key] = value

    chronos_cfg = hf_config.get("chronos_config", {})

    config = {
        "model_type": "chronos",
        "ts_model_class": "ChronosModel",
        "hidden_size": hf_config.get("d_model", 512),
        "num_layers": hf_config.get("num_layers", 8),
        "num_heads": hf_config.get("num_heads", 6),
        "input_format": "tokens",
        "output_format": "token_logits",
        "context_length": chronos_cfg.get("context_length", 512),
        "prediction_length": chronos_cfg.get("prediction_length", 64),
        "d_model": hf_config.get("d_model", 512),
        "d_ff": hf_config.get("d_ff", 1024),
        "d_kv": hf_config.get("d_kv", 64),
        "num_decoder_layers": hf_config.get("num_decoder_layers", 8),
        "vocab_size": hf_config.get("vocab_size", 4096),
        "relative_attention_num_buckets": hf_config.get(
            "relative_attention_num_buckets", 32
        ),
        "relative_attention_max_distance": hf_config.get(
            "relative_attention_max_distance", 128
        ),
        "is_gated_act": hf_config.get("is_gated_act", False),
        "dense_act_fn": hf_config.get("dense_act_fn", "relu"),
        "chronos_config": {
            "tokenizer_class": chronos_cfg.get(
                "tokenizer_class", "MeanScaleUniformBins"
            ),
            "n_tokens": chronos_cfg.get("n_tokens", 4096),
            "n_special_tokens": chronos_cfg.get("n_special_tokens", 2),
            "context_length": chronos_cfg.get("context_length", 512),
            "prediction_length": chronos_cfg.get("prediction_length", 64),
            "num_samples": chronos_cfg.get("num_samples", 20),
            "temperature": chronos_cfg.get("temperature", 1.0),
            "top_k": chronos_cfg.get("top_k", 50),
            "top_p": chronos_cfg.get("top_p", 1.0),
        },
    }

    print(f"  Loaded {len(remapped)} weight tensors")
    return remapped, config


# ---------------------------------------------------------------------------
# Chronos v2 conversion (encoder-only, patch-based)
# ---------------------------------------------------------------------------


def convert_chronos_v2(model_dir: Path) -> tuple[dict[str, mx.array], dict]:
    """Convert Chronos-2 (encoder-only, patch-based) to MLX format."""
    print("Converting Chronos-2 model...")
    weights = load_safetensors_weights(model_dir)
    hf_config = load_hf_config(model_dir)

    # Skip RoPE inv_freq buffers
    remapped = {}
    for key, value in weights.items():
        if "inv_freq" in key:
            continue
        remapped[key] = value

    chronos_cfg = hf_config.get("chronos_config", {})

    config = {
        "model_type": "chronos_v2",
        "ts_model_class": "Chronos2Model",
        "hidden_size": hf_config.get("d_model", 768),
        "num_layers": hf_config.get("num_layers", 12),
        "num_heads": hf_config.get("num_heads", 12),
        "input_format": "patches",
        "output_format": "quantiles",
        "context_length": chronos_cfg.get("context_length", 8192),
        "prediction_length": chronos_cfg.get("max_output_patches", 64)
        * chronos_cfg.get("output_patch_size", 16),
        # T5 encoder config
        "d_model": hf_config.get("d_model", 768),
        "d_ff": hf_config.get("d_ff", 3072),
        "d_kv": hf_config.get("d_kv", 64),
        "vocab_size": hf_config.get("vocab_size", 2),
        "is_gated_act": hf_config.get("is_gated_act", False),
        "dense_act_fn": hf_config.get("dense_act_fn", "relu"),
        "rope_theta": hf_config.get("rope_theta", 10000.0),
        # Chronos-2 specific
        "chronos_config": {
            "context_length": chronos_cfg.get("context_length", 8192),
            "input_patch_size": chronos_cfg.get("input_patch_size", 16),
            "input_patch_stride": chronos_cfg.get("input_patch_stride", 16),
            "output_patch_size": chronos_cfg.get("output_patch_size", 16),
            "max_output_patches": chronos_cfg.get("max_output_patches", 64),
            "quantiles": chronos_cfg.get(
                "quantiles",
                [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            ),
            "use_arcsinh": chronos_cfg.get("use_arcsinh", True),
            "use_reg_token": chronos_cfg.get("use_reg_token", True),
            "time_encoding_scale": chronos_cfg.get("time_encoding_scale", 8192),
        },
    }

    print(f"  Loaded {len(remapped)} weight tensors")
    return remapped, config


# ---------------------------------------------------------------------------
# TimesFM conversion (v2.0 and v2.5)
# ---------------------------------------------------------------------------


def convert_timesfm(model_dir: Path) -> tuple[dict[str, mx.array], dict]:
    """Convert TimesFM (v2.0, v2.5 pytorch, or v2.5 transformers) to MLX format."""
    print("Converting TimesFM model...")
    weights = load_safetensors_weights(model_dir)
    hf_config = load_hf_config(model_dir)

    # Detect transformers variant by presence of decoder.layers keys
    is_transformers = any(k.startswith("decoder.layers.") for k in weights)

    if is_transformers:
        return _convert_timesfm_transformers(weights, hf_config)

    # --- Original pytorch variant (weights kept as-is) ---
    remapped = dict(weights)

    backbone_cfg = hf_config.get("backbone_config", hf_config)
    num_quantiles = len(hf_config.get("quantiles", [0.1]*9))

    patch_length = hf_config.get("patch_length", 32)
    if "tokenizer.hidden_layer.weight" in weights:
        patch_length = weights["tokenizer.hidden_layer.weight"].shape[-1]
        print(f"  Inferred patch_length={patch_length} from tokenizer weights")

    config = {
        "model_type": "timesfm",
        "ts_model_class": "TimesFMModel",
        "hidden_size": backbone_cfg.get("hidden_size", 1280),
        "num_layers": backbone_cfg.get("num_hidden_layers", backbone_cfg.get("num_layers", 20)),
        "num_heads": backbone_cfg.get("num_attention_heads", backbone_cfg.get("num_heads", 16)),
        "intermediate_size": backbone_cfg.get("intermediate_size", 1280),
        "head_dim": backbone_cfg.get("head_dim", 80),
        "patch_length": int(patch_length),
        "quantile_horizon_length": hf_config.get("quantile_horizon_length", 1024),
        "num_quantiles": num_quantiles,
        "context_length": hf_config.get("context_length", 16384),
        "prediction_length": hf_config.get("horizon_length", 128),
        "use_positional_encoding": backbone_cfg.get("use_positional_encoding", False),
        "query_pre_attn_scalar": float(backbone_cfg.get("head_dim", 80)),  # pytorch uses headDim
    }

    print(f"  Loaded {len(remapped)} weight tensors")
    return remapped, config


def _convert_timesfm_transformers(weights: dict, hf_config: dict) -> tuple[dict[str, mx.array], dict]:
    """Convert TimesFM 2.5 transformers variant to MLX format.

    The transformers variant uses separate Q/K/V projections, standard RoPE,
    and key names under decoder.layers.N.* instead of stacked_xf.N.*.
    We fuse Q/K/V and remap to match the existing Swift TimesFMModel layout.
    """
    print("  Detected transformers variant — fusing Q/K/V and remapping keys")
    import re as _re

    num_layers = hf_config.get("num_hidden_layers", 20)
    remapped = {}

    for key, value in weights.items():
        # Tokenizer: decoder.input_ff_layer.* -> tokenizer.*
        if key.startswith("decoder.input_ff_layer."):
            remapped[key.replace("decoder.input_ff_layer.", "tokenizer.")] = value
            continue

        # Transformer layers: decoder.layers.N.* -> stacked_xf.N.*
        if m := _re.match(r"decoder\.layers\.(\d+)\.(.*)", key):
            n, rest = m.group(1), m.group(2)
            # Attention sub-keys
            if rest.startswith("self_attn."):
                attn_rest = rest[len("self_attn."):]
                if attn_rest == "scaling":
                    # Per-dim scale: stacked_xf.N.attn.per_dim_scale.per_dim_scale
                    remapped[f"stacked_xf.{n}.attn.per_dim_scale.per_dim_scale"] = value
                elif attn_rest == "o_proj.weight":
                    remapped[f"stacked_xf.{n}.attn.out.weight"] = value
                elif attn_rest in ("query_ln.weight", "key_ln.weight"):
                    remapped[f"stacked_xf.{n}.attn.{attn_rest}"] = value
                # q/k/v collected separately below
            # MLP sub-keys
            elif rest.startswith("mlp."):
                mlp_rest = rest[len("mlp."):]
                remapped[f"stacked_xf.{n}.{mlp_rest}"] = value
            # Layer norms
            elif rest in ("pre_attn_ln.weight", "post_attn_ln.weight",
                          "pre_ff_ln.weight", "post_ff_ln.weight"):
                remapped[f"stacked_xf.{n}.{rest}"] = value
            continue

        # Output heads and horizon FF — keep as-is
        remapped[key] = value

    # Fuse separate Q, K, V into single qkv_proj per layer
    for i in range(num_layers):
        q_key = f"stacked_xf.{i}.attn.q_proj.weight"
        k_key = f"stacked_xf.{i}.attn.k_proj.weight"
        v_key = f"stacked_xf.{i}.attn.v_proj.weight"
        # The q/k/v keys from decoder.layers.N.self_attn.{q,k,v}_proj.weight
        # were not remapped above — grab them from original weights
        raw_q = weights.get(f"decoder.layers.{i}.self_attn.q_proj.weight")
        raw_k = weights.get(f"decoder.layers.{i}.self_attn.k_proj.weight")
        raw_v = weights.get(f"decoder.layers.{i}.self_attn.v_proj.weight")
        if raw_q is not None and raw_k is not None and raw_v is not None:
            remapped[f"stacked_xf.{i}.attn.qkv_proj.weight"] = mx.concatenate(
                [raw_q, raw_k, raw_v], axis=0)

    # Infer patch_length from tokenizer input dim
    if "tokenizer.hidden_layer.weight" in remapped:
        patch_length = remapped["tokenizer.hidden_layer.weight"].shape[-1]
        print(f"  Inferred patch_length={patch_length} from tokenizer weights")
    else:
        patch_length = hf_config.get("patch_length", 32)

    num_quantiles = len(hf_config.get("quantiles", [0.1]*9))

    config = {
        "model_type": "timesfm",
        "ts_model_class": "TimesFMModel",
        "hidden_size": hf_config.get("hidden_size", 1280),
        "num_layers": hf_config.get("num_hidden_layers", 20),
        "num_heads": hf_config.get("num_attention_heads", 16),
        "intermediate_size": hf_config.get("intermediate_size", 1280),
        "head_dim": hf_config.get("head_dim", 80),
        "patch_length": int(patch_length),
        "quantile_horizon_length": hf_config.get("output_quantile_len", 1024),
        "num_quantiles": num_quantiles,
        "context_length": hf_config.get("context_length", 16384),
        "prediction_length": hf_config.get("horizon_length", 128),
        "use_positional_encoding": False,
        # Transformers variant uses query_pre_attn_scalar instead of headDim for scaling
        "query_pre_attn_scalar": float(hf_config.get("query_pre_attn_scalar", 256.0)),
        "use_rope": True,
        "rope_theta": float(hf_config.get("rope_theta", 10000.0)),
        "use_horizon_ff": False,  # TODO: investigate amplification with horizon_ff_layer
    }

    print(f"  Loaded {len(remapped)} weight tensors (transformers variant)")
    return remapped, config


# ---------------------------------------------------------------------------
# Lag-Llama conversion
# ---------------------------------------------------------------------------


def convert_lag_llama(model_dir: Path) -> tuple[dict[str, mx.array], dict]:
    """Convert Lag-Llama model to MLX format."""
    print("Converting Lag-Llama model...")

    ckpt_path = model_dir / "lag-llama.ckpt"
    if not ckpt_path.exists():
        ckpt_files = list(model_dir.glob("*.ckpt"))
        if ckpt_files:
            ckpt_path = ckpt_files[0]
        else:
            st_files = list(model_dir.glob("*.safetensors"))
            if st_files:
                raw_weights = load_safetensors_weights(model_dir)
            else:
                bin_files = list(model_dir.glob("*.bin"))
                if bin_files:
                    raw_weights = load_pytorch_weights(bin_files[0])
                else:
                    raise FileNotFoundError(
                        f"No checkpoint found in {model_dir}. "
                        "Expected .ckpt, .safetensors, or .bin"
                    )
            ckpt_path = None

    if ckpt_path is not None:
        import pickle
        import torch

        # Lag-Llama checkpoints were saved with an old gluonts version whose
        # modules no longer exist. Use a custom unpickler that stubs missing
        # classes so we can still extract the state_dict and hyper_parameters.
        class _StubClass:
            def __init__(self, *a, **kw): pass
            def __setstate__(self, s): pass
            def __call__(self, *a, **kw): return _StubClass()

        class _SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except Exception:
                    return _StubClass

        class _SafePickleModule:
            Unpickler = _SafeUnpickler
            @staticmethod
            def load(f, **kw): return _SafeUnpickler(f).load()
            loads = pickle.loads
            dump = pickle.dump
            dumps = pickle.dumps
            HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
            DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL

        checkpoint = torch.load(
            str(ckpt_path), map_location="cpu", weights_only=False,
            pickle_module=_SafePickleModule,
        )
        hyper_params = checkpoint.get("hyper_parameters", {})
        model_kwargs = hyper_params.get("model_kwargs", {})
        state_dict = checkpoint.get("state_dict", {})

        raw_weights = {}
        for key, tensor in state_dict.items():
            raw_weights[key] = mx.array(tensor.float().numpy())
    else:
        model_kwargs = {}

    import re as _re

    # Lag-Llama uses a GPT-style naming convention:
    #   model.transformer.wte.*          → input_proj.*
    #   model.transformer.ln_f.scale     → final_norm.weight
    #   model.transformer.h.N.rms_1.scale → layers.N.norm1.weight
    #   model.transformer.h.N.rms_2.scale → layers.N.norm2.weight
    #   model.transformer.h.N.attn.q_proj.* → layers.N.attention.q_proj.*
    #   model.transformer.h.N.attn.kv_proj.weight (fused KV, split 50/50) →
    #       layers.N.attention.k_proj.weight + layers.N.attention.v_proj.weight
    #   model.transformer.h.N.attn.c_proj.* → layers.N.attention.o_proj.*
    #   model.transformer.h.N.mlp.c_fc1.* + c_fc2.* (fused) → layers.N.mlp.gate_up.*
    #   model.transformer.h.N.mlp.c_proj.* → layers.N.mlp.down.*
    #   model.param_proj.proj.0.* → distribution_head.df_proj.*
    #   model.param_proj.proj.1.* → distribution_head.loc_proj.*
    #   model.param_proj.proj.2.* → distribution_head.scale_proj.*
    _head_map = {0: "df_proj", 1: "loc_proj", 2: "scale_proj"}
    _c_fc1: dict = {}
    _c_fc2: dict = {}
    remapped = {}

    for key, value in raw_weights.items():
        k = key[len("model."):] if key.startswith("model.") else key

        if k.startswith("transformer.wte."):
            remapped[k.replace("transformer.wte.", "input_proj.")] = value
        elif k == "transformer.ln_f.scale":
            remapped["final_norm.weight"] = value
        elif m := _re.match(r"transformer\.h\.(\d+)\.(.*)", k):
            n, rest = m.group(1), m.group(2)
            if rest == "rms_1.scale":
                remapped[f"layers.{n}.norm1.weight"] = value
            elif rest == "rms_2.scale":
                remapped[f"layers.{n}.norm2.weight"] = value
            elif rest.startswith("attn.q_proj."):
                remapped[f"layers.{n}.attention.q_proj.{rest[len('attn.q_proj.'):]}"] = value
            elif rest == "attn.kv_proj.weight":
                half = value.shape[0] // 2
                remapped[f"layers.{n}.attention.k_proj.weight"] = value[:half]
                remapped[f"layers.{n}.attention.v_proj.weight"] = value[half:]
            elif rest.startswith("attn.c_proj."):
                remapped[f"layers.{n}.attention.o_proj.{rest[len('attn.c_proj.'):]}"] = value
            elif rest.startswith("mlp.c_fc1."):
                _c_fc1[(n, rest[len("mlp.c_fc1."):])] = value
            elif rest.startswith("mlp.c_fc2."):
                _c_fc2[(n, rest[len("mlp.c_fc2."):])] = value
            elif rest.startswith("mlp.c_proj."):
                remapped[f"layers.{n}.mlp.down.{rest[len('mlp.c_proj.'):]}"] = value
        elif m := _re.match(r"param_proj\.proj\.(\d+)\.(.*)", k):
            idx, suffix = int(m.group(1)), m.group(2)
            if idx in _head_map:
                remapped[f"distribution_head.{_head_map[idx]}.{suffix}"] = value

    # Fuse c_fc1 + c_fc2 → gate_up (SwiGLU gate and up projections)
    for (n, suffix), fc1 in _c_fc1.items():
        if (n, suffix) in _c_fc2:
            remapped[f"layers.{n}.mlp.gate_up.{suffix}"] = mx.concatenate(
                [fc1, _c_fc2[(n, suffix)]], axis=0
            )

    # Infer config from actual weight shapes
    hidden_size = int(model_kwargs.get("n_embd_per_head", 16)) * int(model_kwargs.get("n_head", 9))
    num_layers = int(model_kwargs.get("n_layer", 8))
    num_heads = int(model_kwargs.get("n_head", 9))
    intermediate_size = _c_fc1[list(_c_fc1.keys())[0]].shape[0] if _c_fc1 else hidden_size * 4

    config = {
        "model_type": "lag_llama",
        "ts_model_class": "LagLlamaModel",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "input_format": "lag_features",
        "output_format": "student_t",
        "context_length": model_kwargs.get("context_length", 32),
        "prediction_length": model_kwargs.get("prediction_length", 24),
        "d_model": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_heads,
        "rope_theta": model_kwargs.get("rope_theta", 10000.0),
        "lags_sequence": model_kwargs.get(
            "lags_seq",
            [1, 2, 3, 4, 5, 6, 7, 24, 168, 720, 4320, 8760],
        ),
        "num_parallel_samples": model_kwargs.get("num_parallel_samples", 100),
        "rope_scaling": model_kwargs.get("rope_scaling", None),
    }

    print(f"  Loaded {len(remapped)} weight tensors")
    return remapped, config


# ---------------------------------------------------------------------------
# FlowState conversion (SSM encoder + Legendre decoder)
# ---------------------------------------------------------------------------


def convert_flowstate(model_dir: Path) -> tuple[dict[str, mx.array], dict]:
    """Convert FlowState / granite-flowstate to MLX format."""
    print("Converting FlowState model...")
    weights = load_safetensors_weights(model_dir)
    hf_config = load_hf_config(model_dir)

    # FlowState weights use standard naming — minimal remapping
    remapped = {}
    for key, value in weights.items():
        remapped[key] = value

    config = {
        "model_type": "flowstate",
        "ts_model_class": "FlowStateModel",
        "hidden_size": hf_config.get("embedding_feature_dim", 512),
        "num_layers": hf_config.get("encoder_num_layers", 6),
        "input_format": "raw",
        "output_format": "quantiles",
        "context_length": hf_config.get("context_length", 2048),
        "prediction_length": 128,
        # FlowState-specific
        "embedding_feature_dim": hf_config.get("embedding_feature_dim", 512),
        "encoder_state_dim": hf_config.get("encoder_state_dim", 512),
        "encoder_num_layers": hf_config.get("encoder_num_layers", 6),
        "encoder_num_hippo_blocks": hf_config.get("encoder_num_hippo_blocks", 8),
        "decoder_dim": hf_config.get("decoder_dim", 256),
        "decoder_patch_len": hf_config.get("decoder_patch_len", 24),
        "decoder_type": hf_config.get("decoder_type", "legs"),
        "min_context": hf_config.get("min_context", 2048),
        "init_processing": hf_config.get("init_processing", True),
        "prediction_type": hf_config.get("prediction_type", "quantile"),
        "quantiles": hf_config.get("quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "use_freq": hf_config.get("use_freq", True),
        "with_missing": hf_config.get("with_missing", True),
    }

    print(f"  Loaded {len(remapped)} weight tensors")
    return remapped, config


# ---------------------------------------------------------------------------
# Kairos conversion (MoS-DP transformer)
# ---------------------------------------------------------------------------


def convert_kairos(model_dir: Path) -> tuple[dict[str, mx.array], dict]:
    """Convert Kairos to MLX format."""
    print("Converting Kairos model...")
    weights = load_safetensors_weights(model_dir)
    hf_config = load_hf_config(model_dir)

    # Skip IARoPE inv_freq buffers
    remapped = {}
    for key, value in weights.items():
        if "inv_freq" in key:
            continue
        remapped[key] = value

    config = {
        "model_type": "kairos",
        "ts_model_class": "KairosModel",
        "hidden_size": hf_config.get("d_model", 512),
        "num_layers": hf_config.get("num_layers", 6),
        "num_heads": hf_config.get("num_heads", 8),
        "input_format": "dynamic_patches",
        "output_format": "quantiles",
        "context_length": hf_config.get("context_length", 2048),
        "prediction_length": hf_config.get("prediction_length", 64),
        # Kairos-specific
        "d_model": hf_config.get("d_model", 512),
        "d_ff": hf_config.get("d_ff", 2048),
        "d_kv": hf_config.get("d_kv", 64),
        "num_decoder_layers": hf_config.get("num_decoder_layers", 6),
        "num_decoder_segments": hf_config.get("num_decoder_segments", 2),
        "input_patch_size": hf_config.get("input_patch_size", 128),
        "input_patch_stride": hf_config.get("input_patch_stride", 128),
        "levels": hf_config.get("levels", 3),
        "n_activated_experts": hf_config.get("n_activated_experts", 3),
        "n_null_experts": hf_config.get("n_null_experts", 2),
        "moe_inter_dim": hf_config.get("moe_inter_dim", 1408),
        "position_embedding_type": hf_config.get("position_embedding_type", "instance_wise_rope"),
        "instance_rope_input_feature_dim": hf_config.get("instance_rope_input_feature_dim", 128),
        "is_gated_act": hf_config.get("is_gated_act", False),
        "dense_act_fn": hf_config.get("dense_act_fn", "relu"),
        "quantiles": hf_config.get("quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "use_reg_token": hf_config.get("use_reg_token", True),
    }

    print(f"  Loaded {len(remapped)} weight tensors")
    return remapped, config


# ---------------------------------------------------------------------------
# TiRex conversion (xLSTM / sLSTM)
# ---------------------------------------------------------------------------


def convert_tirex(model_dir: Path) -> tuple[dict[str, mx.array], dict]:
    """Convert TiRex (sLSTM-based) to MLX format."""
    print("Converting TiRex model...")

    # TiRex uses .ckpt format (PyTorch Lightning)
    ckpt_files = list(model_dir.glob("*.ckpt"))
    if ckpt_files:
        import torch

        ckpt_path = ckpt_files[0]
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Extract hyperparameters if available
        hyper_params = checkpoint.get("hyper_parameters", {})

        raw_weights = {}
        for key, tensor in state_dict.items():
            raw_weights[key] = mx.array(tensor.float().numpy())
    else:
        # Try safetensors or .bin
        st_files = list(model_dir.glob("*.safetensors"))
        if st_files:
            raw_weights = load_safetensors_weights(model_dir)
        else:
            bin_files = list(model_dir.glob("*.bin"))
            if bin_files:
                raw_weights = load_pytorch_weights(bin_files[0])
            else:
                raise FileNotFoundError(
                    f"No checkpoint found in {model_dir}. "
                    "Expected .ckpt, .safetensors, or .bin"
                )
        hyper_params = {}

    # Strip "block_stack." prefix (TiRex Lightning convention)
    remapped = {}
    for key, value in raw_weights.items():
        new_key = key
        if new_key.startswith("block_stack."):
            new_key = new_key[len("block_stack."):]
        remapped[new_key] = value

    # Try to infer config from checkpoint hyperparameters or weight shapes
    model_cfg = hyper_params.get("model_config", hyper_params)
    block_kwargs = model_cfg.get("block_kwargs", {})

    # Infer embedding dim from weight shapes
    embedding_dim = 512  # default
    for key, value in remapped.items():
        if key == "out_norm.weight":
            embedding_dim = value.shape[0]
            break

    # Infer num_layers from block indices
    num_layers = 0
    for key in remapped:
        if key.startswith("blocks.") and ".norm_slstm." in key:
            idx = int(key.split(".")[1])
            num_layers = max(num_layers, idx + 1)

    config = {
        "model_type": "tirex",
        "ts_model_class": "TiRexModel",
        "hidden_size": embedding_dim,
        "num_layers": num_layers,
        "input_format": "patches",
        "output_format": "quantiles",
        "context_length": model_cfg.get("train_ctx_len", 2048),
        "prediction_length": 64,
        # TiRex-specific
        "embedding_dim": embedding_dim,
        "input_patch_size": model_cfg.get("input_patch_size", 32),
        "output_patch_size": model_cfg.get("output_patch_size", 32),
        "num_heads": block_kwargs.get("num_heads", 4),
        "ffn_proj_factor": block_kwargs.get("ffn_proj_factor", 2.6667),
        "quantiles": model_cfg.get(
            "quantiles",
            [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98],
        ),
    }

    print(f"  Loaded {len(remapped)} weight tensors")
    return remapped, config


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def quantize_weights(
    weights: dict[str, mx.array], bits: int = 4, group_size: int = 64
) -> tuple[dict[str, mx.array], dict]:
    """Quantize weight tensors using MLX quantization."""
    print(f"  Quantizing to {bits}-bit (group_size={group_size})...")
    quantized = {}
    q_config = {"bits": bits, "group_size": group_size}

    for key, value in weights.items():
        if value.ndim == 2 and "weight" in key and value.shape[-1] % group_size == 0:
            q, scales, biases = mx.quantize(value, group_size=group_size, bits=bits)
            base_key = key.replace(".weight", "")
            quantized[base_key + ".weight"] = q
            quantized[base_key + ".scales"] = scales
            quantized[base_key + ".biases"] = biases
        else:
            quantized[key] = value

    return quantized, q_config


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_model(
    weights: dict[str, mx.array],
    config: dict,
    output_dir: Path,
    q_config: dict | None = None,
):
    """Save weights as safetensors and config as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if q_config:
        config["quantization"] = q_config

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config to {config_path}")

    total_size = sum(v.nbytes for v in weights.values())
    shard_size = 5 * 1024 * 1024 * 1024  # 5 GB

    if total_size <= shard_size:
        out_path = output_dir / "model.safetensors"
        mx.save_safetensors(str(out_path), weights)
        print(f"  Saved weights to {out_path} ({total_size / 1e6:.1f} MB)")
    else:
        shard_idx = 0
        current_shard = {}
        current_size = 0
        for key, value in weights.items():
            if current_size + value.nbytes > shard_size and current_shard:
                out_path = output_dir / f"model-{shard_idx:05d}-of-{99999:05d}.safetensors"
                mx.save_safetensors(str(out_path), current_shard)
                print(f"  Saved shard {shard_idx} ({current_size / 1e6:.1f} MB)")
                shard_idx += 1
                current_shard = {}
                current_size = 0
            current_shard[key] = value
            current_size += value.nbytes
        if current_shard:
            out_path = output_dir / f"model-{shard_idx:05d}-of-{shard_idx:05d}.safetensors"
            mx.save_safetensors(str(out_path), current_shard)
            print(f"  Saved shard {shard_idx} ({current_size / 1e6:.1f} MB)")

        total_shards = shard_idx + 1
        for i in range(total_shards):
            old_name = output_dir / f"model-{i:05d}-of-{99999 if i < shard_idx else shard_idx:05d}.safetensors"
            new_name = output_dir / f"model-{i:05d}-of-{total_shards - 1:05d}.safetensors"
            if old_name.exists() and old_name != new_name:
                old_name.rename(new_name)

    print(f"  Total: {total_size / 1e6:.1f} MB ({len(weights)} tensors)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONVERTERS = {
    "toto": convert_toto,
    "chronos": convert_chronos,
    "chronos_v2": convert_chronos_v2,
    "timesfm": convert_timesfm,
    "lag_llama": convert_lag_llama,
    "flowstate": convert_flowstate,
    "kairos": convert_kairos,
    "tirex": convert_tirex,
}


def generate_readme(
    hf_path: str,
    upload_repo: str,
    model_type: str,
    quantized: bool = False,
    q_bits: int | None = None,
) -> str:
    """Generate a HuggingFace model card README."""
    tags = ["mlx", "time-series", "forecasting", model_type]
    if quantized and q_bits is not None:
        tags.extend(["quantized", f"{q_bits}-bit"])

    tags_yaml = "\n".join(f"- {tag}" for tag in tags)

    return f"""---
library_name: mlx
tags:
{tags_yaml}
base_model: {hf_path}
---

# {upload_repo}

This model was converted from [`{hf_path}`](https://huggingface.co/{hf_path})
using [MLX-Swift-TS](https://github.com/kunal732/MLX-Swift-TS).

## Use with MLX-Swift-TS

```swift
import MLXTimeSeries

let forecaster = try await TimeSeriesForecaster.loadFromHub(id: "{upload_repo}")
let input = TimeSeriesInput.univariate(historicalValues)
let prediction = forecaster.forecast(input: input, predictionLength: 64)
```

## Original Model

[{hf_path}](https://huggingface.co/{hf_path})
"""


def upload_to_hub(mlx_path: str, upload_repo: str):
    """Upload a converted model directory to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(folder_path=mlx_path, repo_id=upload_repo)
    print(f"  Uploaded to https://huggingface.co/{upload_repo}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert time series models to MLX safetensors format."
    )
    parser.add_argument(
        "--hf-path",
        required=True,
        help="HuggingFace model ID (e.g. Datadog/Toto-Open-Base-1.0) "
        "or local directory path.",
    )
    parser.add_argument(
        "--mlx-path",
        default="mlx_model",
        help="Output directory for converted model (default: mlx_model).",
    )
    parser.add_argument(
        "--model-type",
        choices=list(CONVERTERS.keys()),
        default=None,
        help="Model type (auto-detected from model ID if not specified).",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Quantize weights after conversion.",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        choices=[2, 4, 8],
        help="Quantization bits (default: 4).",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64).",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Target dtype for non-quantized weights (default: float16).",
    )
    parser.add_argument(
        "--upload-repo",
        default=None,
        help="HuggingFace repo ID to upload the converted model "
        "(e.g. mlx-community/Toto-Open-Base-1.0-4bit).",
    )
    args = parser.parse_args()

    model_type = args.model_type or detect_model_type(args.hf_path)
    print(f"Model type: {model_type}")

    model_path = Path(args.hf_path)
    if not model_path.exists():
        print(f"Downloading {args.hf_path} from HuggingFace...")
        from huggingface_hub import snapshot_download

        model_path = Path(
            snapshot_download(
                args.hf_path,
                allow_patterns=["*.safetensors", "*.ckpt", "*.bin", "config.json"],
            )
        )
        print(f"  Downloaded to {model_path}")

    converter = CONVERTERS[model_type]
    weights, config = converter(model_path)

    dtype_map = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    target_dtype = dtype_map[args.dtype]
    for key in weights:
        if weights[key].dtype in (mx.float32, mx.float16, mx.bfloat16):
            weights[key] = weights[key].astype(target_dtype)

    q_config = None
    if args.quantize:
        weights, q_config = quantize_weights(
            weights, bits=args.q_bits, group_size=args.q_group_size
        )

    output_dir = Path(args.mlx_path)
    save_model(weights, config, output_dir, q_config)
    print(f"\nDone! Converted model saved to {output_dir}")

    if args.upload_repo:
        readme = generate_readme(
            hf_path=args.hf_path,
            upload_repo=args.upload_repo,
            model_type=model_type,
            quantized=args.quantize,
            q_bits=args.q_bits if args.quantize else None,
        )
        readme_path = output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme)
        print(f"  Generated {readme_path}")

        upload_to_hub(str(output_dir), args.upload_repo)


if __name__ == "__main__":
    main()
