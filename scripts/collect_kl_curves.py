#!/usr/bin/env python3
"""
Collect KL(beta/epsilon) curves for steering and isotropic perturbations.

Usage examples:

  - Per-chunk steering vectors (per example/chunk):
      python scripts/collect_kl_curves.py \
        --mode steer --steer-type per-chunk \
        --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

  - Centered steering vectors (z_i - mean_{j!=i} z_j):
      python scripts/collect_kl_curves.py \
        --mode steer --steer-type centered \
        --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

  - Global diff-in-means steering vector:
      python scripts/collect_kl_curves.py \
        --mode steer --steer-type diff-in-means --importance-threshold 0.2 \
        --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

  - Isotropic random perturbations at anchor span:
      python scripts/collect_kl_curves.py \
        --mode perturb --epsilons 0 0.5 1 2 5 10 \
        --n-directions 32 --layers -1 \
        --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

Saves JSON artifacts into data/:
  - Steering:  data/kl_curves_steer_{steer_type}_{model_tag}.json
  - Perturb:   data/kl_curves_perturb_{model_tag}.json

Notes:
  - Uses teacher-forced spans at the anchor chunk as in the notebooks.
  - Scales deltas by local residual RMS at the intervention position.
  - Assumes annotated data and chunk activations exist under data/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

# Ensure repo root on path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Proactively disable/short-circuit accelerate to avoid circular-import issues
# in some accelerate>=1.2.x versions when transformers imports generation utils.
# We don't rely on accelerate in this script, so a lightweight stub suffices.
def _disable_accelerate_import():
    """Install a minimal accelerate stub with a valid __spec__ to satisfy
    transformers' availability checks and avoid circular imports."""
    try:
        import sys as _sys, types as _types
        from importlib.machinery import ModuleSpec
        os.environ.setdefault('TRANSFORMERS_NO_ACCELERATE', '1')
        # Prefer importing the real accelerate if available; only synthesize if import fails
        _real_accelerate = None
        try:
            import importlib as _importlib
            _real_accelerate = _sys.modules.get('accelerate') or _importlib.import_module('accelerate')
        except Exception:
            _real_accelerate = None

        if _real_accelerate is None:
            pkg = _types.ModuleType('accelerate')
            pkg.__path__ = []  # mark as package
            pkg.__spec__ = ModuleSpec(name='accelerate', loader=None, is_package=True)

            hooks = _types.ModuleType('accelerate.hooks')
            hooks.__spec__ = ModuleSpec(name='accelerate.hooks', loader=None, is_package=False)

            class _AlignDevicesHook:
                def __init__(self, *a, **k):
                    pass

            def _add_hook_to_module(*a, **k):
                return None
            def _remove_hook_from_module(*a, **k):
                return None

            hooks.AlignDevicesHook = _AlignDevicesHook
            hooks.add_hook_to_module = _add_hook_to_module
            hooks.remove_hook_from_module = _remove_hook_from_module

            # Minimal no-op dispatch_model to satisfy `from accelerate import dispatch_model`
            def _dispatch_model(model, *a, **k):
                return model

            # Expose a minimal utils submodule as well (some versions import from accelerate.utils)
            utils = _types.ModuleType('accelerate.utils')
            utils.__spec__ = ModuleSpec(name='accelerate.utils', loader=None, is_package=False)
            utils.dispatch_model = _dispatch_model
            # Also provide newer helpers some Transformers versions import
            def _get_balanced_memory(*a, **k):
                return {}
            def _infer_auto_device_map(*a, **k):
                return {}
            def _reduce(*a, **k):
                # passthrough placeholder used in some detection losses
                return a[0] if a else None
            # Minimal init_empty_weights context manager
            def _init_empty_weights(*a, **k):
                class _CM:
                    def __enter__(self):
                        return None
                    def __exit__(self, exc_type, exc, tb):
                        return False
                return _CM()
            # Other utils some TF versions expect
            def _check_tied_parameters_on_same_device(*a, **k):
                return True
            def _find_tied_parameters(*a, **k):
                return {}
            def _is_compiled_module(*a, **k):
                return False
            def _is_deepspeed_available(*a, **k):
                return False
            def _is_torchdynamo_compiling(*a, **k):
                return False
            def _extract_model_from_parallel(model, *a, **k):
                return model
            def _get_max_memory(*a, **k):
                return {}
            def _load_offloaded_weights(*a, **k):
                return None
            utils.get_balanced_memory = _get_balanced_memory
            utils.infer_auto_device_map = _infer_auto_device_map
            utils.reduce = _reduce
            utils.check_tied_parameters_on_same_device = _check_tied_parameters_on_same_device
            utils.find_tied_parameters = _find_tied_parameters
            utils.is_compiled_module = _is_compiled_module
            utils.is_deepspeed_available = _is_deepspeed_available
            utils.is_torchdynamo_compiling = _is_torchdynamo_compiling
            utils.extract_model_from_parallel = _extract_model_from_parallel
            utils.get_max_memory = _get_max_memory
            utils.load_offloaded_weights = _load_offloaded_weights
            # Provide PartialState on the top-level accelerate module
            class _PartialState:
                def __init__(self, *a, **k):
                    pass
                def __getattr__(self, name):
                    raise AttributeError(name)

            pkg.dispatch_model = _dispatch_model
            pkg.hooks = hooks
            pkg.dispatch_model = _dispatch_model
            pkg.PartialState = _PartialState
            pkg.infer_auto_device_map = _infer_auto_device_map
            pkg.init_empty_weights = _init_empty_weights
            _sys.modules['accelerate'] = pkg
            _sys.modules['accelerate.hooks'] = hooks
            _sys.modules['accelerate.utils'] = utils
        else:
            # Patch the real accelerate module with missing symbols expected by Transformers
            mod = _sys.modules.get('accelerate') or _real_accelerate
            if getattr(mod, '__spec__', None) is None:
                mod.__spec__ = ModuleSpec(name='accelerate', loader=None, is_package=True)
            # Ensure hooks submodule exists with spec
            if 'accelerate.hooks' not in _sys.modules:
                hooks = _types.ModuleType('accelerate.hooks')
                hooks.__spec__ = ModuleSpec(name='accelerate.hooks', loader=None, is_package=False)
                _sys.modules['accelerate.hooks'] = hooks
            if not hasattr(_sys.modules['accelerate.hooks'], 'remove_hook_from_module'):
                def _remove_hook_from_module(*a, **k):
                    return None
                setattr(_sys.modules['accelerate.hooks'], 'remove_hook_from_module', _remove_hook_from_module)
            # Ensure utils submodule exists and exposes required symbols
            try:
                utils = _sys.modules.get('accelerate.utils') or _importlib.import_module('accelerate.utils')
            except Exception:
                # As a last resort, synthesize a minimal module
                utils = _types.ModuleType('accelerate.utils')
                utils.__spec__ = ModuleSpec(name='accelerate.utils', loader=None, is_package=False)
                _sys.modules['accelerate.utils'] = utils
            def _ensure(attr, fn):
                if not hasattr(utils, attr):
                    setattr(utils, attr, fn)
            # Patch in missing helpers expected by Transformers integrations
            def _dispatch_model(model, *a, **k):
                return model
            def _get_balanced_memory(*a, **k):
                return {}
            def _infer_auto_device_map(*a, **k):
                return {}
            def _reduce(*a, **k):
                return a[0] if a else None
            def _init_empty_weights(*a, **k):
                class _CM:
                    def __enter__(self):
                        return None
                    def __exit__(self, exc_type, exc, tb):
                        return False
                return _CM()
            def _check_tied_parameters_on_same_device(*a, **k):
                return True
            def _find_tied_parameters(*a, **k):
                return {}
            def _is_compiled_module(*a, **k):
                return False
            def _is_deepspeed_available(*a, **k):
                return False
            def _is_torchdynamo_compiling(*a, **k):
                return False
            def _extract_model_from_parallel(model, *a, **k):
                return model
            def _get_max_memory(*a, **k):
                return {}
            def _load_offloaded_weights(*a, **k):
                return None
            class _PartialState:
                def __init__(self, *a, **k):
                    pass
                def __getattr__(self, name):
                    raise AttributeError(name)
            if not hasattr(mod, 'dispatch_model'):
                setattr(mod, 'dispatch_model', _dispatch_model)
            if not hasattr(mod, 'PartialState'):
                setattr(mod, 'PartialState', _PartialState)
            if not hasattr(mod, 'infer_auto_device_map'):
                setattr(mod, 'infer_auto_device_map', _infer_auto_device_map)
            if not hasattr(mod, 'init_empty_weights'):
                setattr(mod, 'init_empty_weights', _init_empty_weights)
            _ensure('dispatch_model', _dispatch_model)
            _ensure('get_balanced_memory', _get_balanced_memory)
            _ensure('infer_auto_device_map', _infer_auto_device_map)
            _ensure('reduce', _reduce)
            _ensure('check_tied_parameters_on_same_device', _check_tied_parameters_on_same_device)
            _ensure('find_tied_parameters', _find_tied_parameters)
            _ensure('is_compiled_module', _is_compiled_module)
            _ensure('is_deepspeed_available', _is_deepspeed_available)
            _ensure('is_torchdynamo_compiling', _is_torchdynamo_compiling)
            _ensure('extract_model_from_parallel', _extract_model_from_parallel)
            _ensure('get_max_memory', _get_max_memory)
            _ensure('load_offloaded_weights', _load_offloaded_weights)
    except Exception:
        pass

_disable_accelerate_import()


def _repo_root() -> Path:
    # Scripts assume Python path includes repo root; prefer cwd parent with data/
    cwd = Path.cwd().resolve()
    if (cwd / 'data').exists():
        return cwd
    if cwd.parent and (cwd.parent / 'data').exists():
        return cwd.parent
    return cwd


def _device_from_torch() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return 'cuda'
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def _model_tag(name: str) -> str:
    return name.replace('/', '-')


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)
    print(f"Saved: {path}")


def _load_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def unit_random_direction(dim: int, device: str = 'cpu'):
    import torch
    v = torch.randn(dim, device=device, dtype=torch.float32)
    return v / (v.norm() + 1e-12)


def _get_examples(annotated_payload: dict, max_examples: Optional[int] = None) -> List[dict]:
    # Support both legacy dict payloads with {'examples': [...]} and plain list payloads
    if isinstance(annotated_payload, dict):
        src = annotated_payload.get('examples', [])
    else:
        src = annotated_payload or []
    exs = [ex for ex in src if isinstance(ex, dict) and ex.get('thought_anchor_sentence')]
    if max_examples is not None:
        exs = exs[: int(max_examples)]
    return exs


@dataclass
class CurveRec:
    example_index: int
    chunk_index: int
    importance: float
    ys: List[float]
    layer: Optional[int] = None  # for perturbation outputs


def collect_steer_curves(
    *,
    model_name: str,
    steer_type: str,
    betas: np.ndarray,
    max_examples: Optional[int],
    device: str,
    repo_root: Path,
) -> dict:
    from utils import (
        load_model_and_vectors,
        compute_kl_curve_for_chunk,
    )
    import torch
    # Try nnsight loader; fallback to plain HF if import/runtime fails
    try:
        model, tokenizer, _ = load_model_and_vectors(model_name=model_name, compute_features=False, device=device)
    except Exception as e:
        print("load_model_and_vectors failed; falling back to HF loader:", e)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch_dtype = torch.bfloat16 if (device.startswith('cuda') and torch.cuda.is_available()) else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device if device!='cpu' else None)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    tag = _model_tag(model_name)
    annotated_path = repo_root / 'data' / f'annotated_data_{tag}.json'
    activations_path = repo_root / 'data' / f'chunk_activations_{tag}.json'
    annotated = _load_json(annotated_path)
    anchors_payload = _load_json(activations_path)

    examples = _get_examples(annotated, max_examples)
    examples_anchors = anchors_payload.get('examples', [])

    curves: List[CurveRec] = []

    if steer_type == 'per-chunk':
        for ex_i, ex in enumerate(tqdm(examples, desc="Steer examples", unit="ex")):
            if ex_i >= len(examples_anchors):
                break
            anchors_ex = examples_anchors[ex_i] or {}
            layer_idx = anchors_ex.get('layer', model.config.num_hidden_layers - 1)
            # For each chunk in this example, compute KL(beta) curve using its own vector
            for ch in tqdm(anchors_ex.get('chunks', []), desc=f"Chunks ex {ex_i}", unit="chunk", leave=False):
                idx = int(ch.get('chunk_index', 0))
                ys = compute_kl_curve_for_chunk(
                    model,
                    tokenizer,
                    ex,
                    anchors_ex,
                    layer_idx=int(layer_idx),
                    betas=betas,
                    device=device,
                    chunk_index=idx,
                )
                imp = 0.0
                try:
                    imp = float(ex.get('counterfactual_importance_kl', [0.0])[idx])
                except Exception:
                    pass
                if ys:
                    curves.append(CurveRec(example_index=ex_i, chunk_index=idx, importance=imp, ys=[float(y) for y in ys]))

    elif steer_type == 'centered':
        for ex_i, ex in enumerate(tqdm(examples, desc="Steer examples", unit="ex")):
            if ex_i >= len(examples_anchors):
                break
            anchors_ex = examples_anchors[ex_i] or {}
            layer_idx = anchors_ex.get('layer', model.config.num_hidden_layers - 1)
            # Build per-chunk base vectors z_i
            base_chunk_vecs: Dict[int, torch.Tensor] = {}
            for ch in anchors_ex.get('chunks', []):
                vec = torch.tensor(ch.get('vector', []), dtype=torch.float32, device=device)
                base_chunk_vecs[int(ch.get('chunk_index', 0))] = vec
            if not base_chunk_vecs:
                continue
            all_idxs = sorted(base_chunk_vecs.keys())
            stack = torch.stack([base_chunk_vecs[i].detach().to(torch.float32) for i in all_idxs], dim=0)
            # v_i = unit(z_i - mean_{j!=i} z_j)
            centered_map: Dict[int, List[float]] = {}
            for k, i in enumerate(all_idxs):
                z_i = stack[k]
                if len(all_idxs) > 1:
                    mean_others = (stack.sum(dim=0) - z_i) / (len(all_idxs) - 1)
                else:
                    mean_others = torch.zeros_like(z_i)
                diff = z_i - mean_others
                diff = diff / (diff.norm() + 1e-12)
                centered_map[i] = [float(x) for x in diff.detach().to('cpu').numpy().tolist()]
            # Build a faux anchors_ex with centered vectors
            centered_ex = {
                'layer': int(layer_idx),
                'chunks': [{'chunk_index': int(i), 'vector': centered_map[int(i)]} for i in all_idxs],
            }
            # Compute KL curves using centered vectors
            for idx in tqdm(all_idxs, desc=f"Chunks ex {ex_i}", unit="chunk", leave=False):
                ys = compute_kl_curve_for_chunk(
                    model,
                    tokenizer,
                    ex,
                    centered_ex,
                    layer_idx=int(layer_idx),
                    betas=betas,
                    device=device,
                    chunk_index=int(idx),
                )
                imp = 0.0
                try:
                    imp = float(ex.get('counterfactual_importance_kl', [0.0])[idx])
                except Exception:
                    pass
                if ys:
                    curves.append(CurveRec(example_index=ex_i, chunk_index=int(idx), importance=imp, ys=[float(y) for y in ys]))

    elif steer_type == 'diff-in-means':
        # Build a global difference-in-means vector from all examples/chunks
        pos_vecs = []
        neg_vecs = []
        # Use threshold defined by caller (passed via env var or default). We'll compute here after parsing args.
        raise RuntimeError("diff-in-means requires importance_threshold; handled in driver")
    else:
        raise ValueError(f"Unknown steer_type: {steer_type}")

    return {
        'model': model_name,
        'mode': 'steer',
        'steer_type': steer_type,
        'betas': [float(b) for b in betas.tolist()],
        'curves': [asdict(c) for c in curves],
    }


def collect_steer_curves_diff_in_means(
    *,
    model_name: str,
    betas: np.ndarray,
    max_examples: Optional[int],
    device: str,
    repo_root: Path,
    importance_threshold: float,
) -> dict:
    from utils import (
        load_model_and_vectors,
        split_solution_into_chunks,
        compute_kl_curve_for_chunk,
    )
    import torch

    try:
        model, tokenizer, _ = load_model_and_vectors(model_name=model_name, compute_features=False, device=device)
    except Exception as e:
        print("load_model_and_vectors failed; falling back to HF loader:", e)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch_dtype = torch.bfloat16 if (device.startswith('cuda') and torch.cuda.is_available()) else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device if device!='cpu' else None)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    tag = _model_tag(model_name)
    annotated_path = repo_root / 'data' / f'annotated_data_{tag}.json'
    activations_path = repo_root / 'data' / f'chunk_activations_{tag}.json'
    annotated = _load_json(annotated_path)
    anchors_payload = _load_json(activations_path)

    examples = _get_examples(annotated, max_examples)
    examples_anchors = anchors_payload.get('examples', [])

    # Gather positive/negative sets by threshold
    pos_vecs = []
    neg_vecs = []
    for ex_i, ex in enumerate(tqdm(examples, desc="Gather vectors", unit="ex")):
        if ex_i >= len(examples_anchors):
            break
        anchors_ex = examples_anchors[ex_i] or {}
        imps = ex.get('counterfactual_importance_kl', []) or []
        for ch in anchors_ex.get('chunks', []):
            idx = int(ch.get('chunk_index', 0))
            vec = np.array(ch.get('vector', []), dtype=np.float32)
            imp = float(imps[idx]) if idx < len(imps) else 0.0
            if imp >= importance_threshold:
                pos_vecs.append(vec)
            else:
                neg_vecs.append(vec)
    if not pos_vecs or not neg_vecs:
        raise RuntimeError("Not enough vectors to compute diff-in-means (check threshold and data)")
    m_pos = np.stack(pos_vecs, axis=0).mean(axis=0)
    m_neg = np.stack(neg_vecs, axis=0).mean(axis=0)
    g = m_pos - m_neg
    g = g / (np.linalg.norm(g) + 1e-12)
    global_vec = g.astype(np.float32).tolist()

    curves: List[CurveRec] = []
    # Use this global vector for every example/chunk, but apply at that example's layer
    for ex_i, ex in enumerate(tqdm(examples, desc="Diff-in-means examples", unit="ex")):
        if ex_i >= len(examples_anchors):
            break
        anchors_ex = examples_anchors[ex_i] or {}
        layer_idx = anchors_ex.get('layer', model.config.num_hidden_layers - 1)
        # Create a synthetic anchors_ex mapping that repeats the global vector for all chunks
        try:
            cot_text = ex.get('cot') or ''
            chunks = split_solution_into_chunks(cot_text)
        except Exception:
            chunks = []
        if not chunks:
            continue
        synth_ex = {
            'layer': int(layer_idx),
            'chunks': [{'chunk_index': int(i), 'vector': global_vec} for i in range(len(chunks))],
        }
        for idx in tqdm(range(len(chunks)), desc=f"Chunks ex {ex_i}", unit="chunk", leave=False):
            ys = compute_kl_curve_for_chunk(
                model,
                tokenizer,
                ex,
                synth_ex,
                layer_idx=int(layer_idx),
                betas=betas,
                device=device,
                chunk_index=int(idx),
            )
            imp = 0.0
            try:
                imp = float(ex.get('counterfactual_importance_kl', [0.0])[idx])
            except Exception:
                pass
            if ys:
                curves.append(CurveRec(example_index=ex_i, chunk_index=int(idx), importance=imp, ys=[float(y) for y in ys]))

    return {
        'model': model_name,
        'mode': 'steer',
        'steer_type': 'diff-in-means',
        'importance_threshold': float(importance_threshold),
        'betas': [float(b) for b in betas.tolist()],
        'global_vector': global_vec,
        'curves': [asdict(c) for c in curves],
    }


def collect_perturb_curves(
    *,
    model_name: str,
    epsilons: np.ndarray,
    n_directions: int,
    layers: Optional[List[int]],
    max_examples: Optional[int],
    device: str,
    repo_root: Path,
) -> dict:
    from utils import (
        load_model_and_vectors,
        split_solution_into_chunks,
        forward_with_logits as _fw,
        kl_from_logits as _kl,
        find_chunk_start_token as _find_span,
    )
    import torch

    try:
        model, tokenizer, _ = load_model_and_vectors(model_name=model_name, compute_features=False, device=device)
    except Exception as e:
        print("load_model_and_vectors failed; falling back to HF loader:", e)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch_dtype = torch.bfloat16 if (device.startswith('cuda') and torch.cuda.is_available()) else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device if device!='cpu' else None)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    hidden_size = int(getattr(model.config, 'hidden_size', getattr(model.config, 'n_embd', 0)))
    num_layers = int(getattr(model.config, 'num_hidden_layers', 1))
    if not layers:
        layers = [num_layers - 1]
    layers = [num_layers - 1 if int(L) < 0 else int(L) for L in layers]

    tag = _model_tag(model_name)
    annotated_path = repo_root / 'data' / f'annotated_data_{tag}.json'
    annotated = _load_json(annotated_path)
    examples = _get_examples(annotated, max_examples)

    @torch.no_grad()
    def _forward_with_logits(input_ids, attention_mask):
        return _fw(model, input_ids=input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def logits_with_random_perturb_full(input_ids, attention_mask, eps: float, layer_idx: int, dir_vec, target_pos: int):
        backbone = getattr(model.model, 'model', model.model)
        target = backbone.layers[int(layer_idx)]
        d = dir_vec.to(torch.float32)
        d = d / (d.norm() + 1e-12)

        def hook(module, inputs, output):
            out = output
            try:
                if isinstance(out, tuple):
                    h = out[0].clone(); rest = out[1:]
                else:
                    h = out.clone(); rest = tuple()
                pos = int(target_pos)
                if pos < 0 or pos >= h.shape[1]:
                    return output
                h_slice_fp32 = h[:, pos:pos+1, :].to(torch.float32)
                d_local = d.to(device=h_slice_fp32.device, dtype=torch.float32)
                rms = torch.sqrt(torch.mean(h_slice_fp32 ** 2) + 1e-20)
                delta_fp32 = (float(eps) * rms) * d_local.view(1, 1, -1)
                h[:, pos:pos+1, :] = (h_slice_fp32 + delta_fp32).to(h.dtype)
                return (h,) + rest if isinstance(out, tuple) else h
            except Exception:
                return output

        handle = target.register_forward_hook(hook)
        try:
            return _forward_with_logits(input_ids, attention_mask)
        finally:
            handle.remove()

    curves: List[CurveRec] = []

    for ex_i, ex in enumerate(tqdm(examples, desc="Perturb examples", unit="ex")):
        question = ex.get('prompt', '')
        cot_text = ex.get('cot') or ''
        try:
            chunks = split_solution_into_chunks(cot_text)
        except Exception:
            import re
            chunks = [p.strip() for p in re.split(r'(?<=[\.\!\?])\s+|\n\n+', cot_text) if p.strip()]
        if not chunks:
            continue

        for idx, chunk_text in enumerate(tqdm(chunks, desc=f"Chunks ex {ex_i}", unit="chunk", leave=False)):
            prefix_text = '\n'.join(chunks[:idx])
            ids_pref, am_pref, ids_full, am_full, s_idx, n_steps = _find_span(tokenizer, question, prefix_text, chunk_text, device)
            logits_full = _forward_with_logits(ids_full, am_full)
            seq_len = int(logits_full.shape[1])
            start = max(0, int(s_idx) - 1)
            n_eff = max(0, min(int(n_steps), int(seq_len - start)))
            if n_eff == 0:
                continue
            base_span = logits_full[:, start:start+n_eff, :]
            base_span_cpu = base_span.to(torch.float32).cpu()
            del logits_full, base_span
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            imp = 0.0
            try:
                imp = float(ex.get('counterfactual_importance_kl', [0.0]*len(chunks))[idx])
            except Exception:
                pass

            for L in layers:
                y_curve: List[float] = []
                for eps in epsilons:
                    if abs(float(eps)) < 1e-12:
                        y_curve.append(0.0)
                        continue
                    dir_kls = []
                    for _ in range(int(n_directions)):
                        d = unit_random_direction(hidden_size, device=device)
                        steered_logits = logits_with_random_perturb_full(ids_full, am_full, float(eps), int(L), d, int(start))
                        steered_span = steered_logits[:, start:start+n_eff, :]
                        steered_span_cpu = steered_span.to(torch.float32).cpu()
                        del steered_logits, steered_span
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        V = steered_span_cpu.shape[-1]
                        kl_t = _kl(steered_span_cpu.reshape(-1, V), base_span_cpu.reshape(-1, V))
                        dir_kls.append(float(kl_t.mean().item()))
                    y_curve.append(sum(dir_kls)/len(dir_kls) if dir_kls else 0.0)
                curves.append(CurveRec(example_index=ex_i, chunk_index=idx, importance=imp, ys=y_curve, layer=int(L)))

    return {
        'model': model_name,
        'mode': 'perturb',
        'epsilons': [float(e) for e in epsilons.tolist()],
        'n_directions': int(n_directions),
        'layers': [int(L) for L in layers],
        'curves': [asdict(c) for c in curves],
    }


def main():
    parser = argparse.ArgumentParser(description="Collect KL curves for steering or perturbations")
    parser.add_argument('--mode', choices=['steer', 'perturb'], required=True)
    parser.add_argument('--model-name', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    parser.add_argument('--max-examples', type=int, default=None)
    # Steering args
    parser.add_argument('--steer-type', choices=['per-chunk', 'centered', 'diff-in-means'])
    parser.add_argument('--betas', type=float, nargs='*', default=None, help='Steering amplitudes (× RMS)')
    parser.add_argument('--importance-threshold', type=float, default=0.2, help='Threshold for diff-in-means split')
    # Perturbation args
    parser.add_argument('--epsilons', type=float, nargs='*', default=None, help='Perturbation magnitudes (× RMS)')
    parser.add_argument('--n-directions', type=int, default=16)
    parser.add_argument('--layers', type=int, nargs='*', default=None, help='Layer indices (use -1 for last)')

    args = parser.parse_args()
    repo_root = _repo_root()
    device = _device_from_torch()

    if args.mode == 'steer':
        steer_type = args.steer_type or 'per-chunk'
        betas = np.asarray(args.betas if args.betas is not None else np.linspace(-10, 10, 21), dtype=float)
        if steer_type == 'diff-in-means':
            payload = collect_steer_curves_diff_in_means(
                model_name=args.model_name,
                betas=betas,
                max_examples=args.max_examples,
                device=device,
                repo_root=repo_root,
                importance_threshold=float(args.importance_threshold),
            )
        else:
            payload = collect_steer_curves(
                model_name=args.model_name,
                steer_type=steer_type,
                betas=betas,
                max_examples=args.max_examples,
                device=device,
                repo_root=repo_root,
            )
        out = repo_root / 'data' / f"kl_curves_steer_{steer_type}_{_model_tag(args.model_name)}.json"
        _save_json(out, payload)

    elif args.mode == 'perturb':
        eps = np.asarray(args.epsilons if args.epsilons is not None else np.linspace(0, 10, 21), dtype=float)
        payload = collect_perturb_curves(
            model_name=args.model_name,
            epsilons=eps,
            n_directions=int(args.n_directions),
            layers=args.layers,
            max_examples=args.max_examples,
            device=device,
            repo_root=repo_root,
        )
        out = repo_root / 'data' / f"kl_curves_perturb_{_model_tag(args.model_name)}.json"
        _save_json(out, payload)

    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == '__main__':
    main()
