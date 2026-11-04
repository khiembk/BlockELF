#!/usr/bin/env python
"""Evaluate a block-revelation checkpoint with unconditional block-to-block (Algorithm 2)
generation — the unconditional counterpart of eval_block.py (used for WMT14/XSum).

Drives generation through `utils.generation_utils.generate_block_uncond` — the same
fixed-b committed-context revelation loop the model was trained for (no source; the
full canvas starts out fully "active") — then scores the samples with the same
Gen. PPL / entropy metric `test_generation_uncond` uses for OWT. Single-device, eager
Python loop (no pmap): intended for checking in on a checkpoint's quality, not
high-throughput batch eval.
"""
import argparse
import copy
import json
import logging
import os
import sys

import jax
try:
    jax.distributed.initialize()
except (RuntimeError, ValueError):
    pass

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax.numpy as jnp
import optax
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder
from modules.model import ELF_models
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import load_encoder_checkpoint, load_checkpoint
from utils.train_utils import TrainState, get_optimizer
from utils.generation_utils import generate_block_uncond
from utils.metrics_utils import Metrics
from configs.config import load_config_from_yaml, apply_config_overrides, load_sampling_configs

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ELF block-revelation checkpoint via unconditional block-to-block generation")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--config_override", action="append", default=[])
    p.add_argument("--checkpoint_path", type=str, required=True,
                    help="Checkpoint file/dir (dir resolves to its latest checkpoint_N).")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--num_sampling_steps", type=int, default=None,
                    help="Override sampling_config.num_sampling_steps[0] (faster eval with fewer steps).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_examples", type=int, default=5)
    p.add_argument("--out_path", type=str, default=None,
                    help="Optional path to write generated samples as JSONL.")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)
    sampling_config = config.sampling_configs[0]
    if args.num_sampling_steps is not None:
        sampling_config.num_sampling_steps = [args.num_sampling_steps]

    log_for_0(f"Checkpoint: {args.checkpoint_path}")
    log_for_0(f"block_size={config.block_size}, use_block_revelation={config.use_block_revelation}")
    log_for_0(f"sampling_config={sampling_config}")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)

    encoder_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    vocab_size = len(tokenizer)
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens, num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size, num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )
    input_dim = 2 * encoder_config.d_model if config.self_cond_prob > 0 else encoder_config.d_model
    dummy_x = jnp.ones((1, config.max_length, input_dim))
    dummy_self_cond_cfg_scale = jnp.ones((1,)) if config.num_self_cond_cfg_tokens > 0 else None
    elf_params = model.init(init_rng, x=dummy_x, t=jnp.ones((1,)), deterministic=True,
                             self_cond_cfg_scale=dummy_self_cond_cfg_scale)

    # Must match train.py's optimizer exactly (same `optimizer: muon/adamw` config) so
    # the opt_state pytree structure matches the checkpoint being restored.
    lr_schedule = optax.constant_schedule(1e-4)
    optimizer = get_optimizer(config, lr_schedule, grad_accum_steps=config.grad_accum_steps)
    state = TrainState.create(
        apply_fn=model.apply, params=elf_params["params"], tx=optimizer,
        dropout_rng=dropout_rng, ema_params1=copy.deepcopy(elf_params["params"]),
    )

    state, loaded_step = load_checkpoint(args.checkpoint_path, state)
    log_for_0(f"Loaded checkpoint at step {loaded_step}")

    ppl_metrics = Metrics(
        gen_ppl_eval_model_name_or_path=config.eval_ppl_model,
        eval_ppl_batch_size=config.eval_ppl_batch_size,
        eval_context_size=config.eval_ppl_max_length,
    )

    hypotheses, printed = [], 0
    samples_done = 0
    while samples_done < args.num_samples:
        cur_batch = min(args.batch_size, args.num_samples - samples_done)
        rng, gen_rng = jax.random.split(rng)
        texts = generate_block_uncond(
            state=state, encoder_params=encoder_params, encoder_apply_fn=encoder_model.apply,
            tokenizer=tokenizer, config=config, sampling_config=sampling_config,
            batch_size=cur_batch, rng=gen_rng,
        )
        hypotheses.extend(texts)
        samples_done += len(texts)

        while printed < args.print_examples and printed < len(hypotheses):
            log_for_0(f"[{printed}] GEN: {hypotheses[printed]}")
            printed += 1

        log_for_0(f"Generated {samples_done}/{args.num_samples}...")

    empty_count = sum(1 for h in hypotheses if not h.strip())
    nonempty = [h for h in hypotheses if h.strip()]

    ppl_results = None
    if nonempty:
        ppl_results = ppl_metrics.record_generative_perplexity(
            text_samples=nonempty, max_length=config.eval_ppl_max_length, retokenize=True,
        )

    log_for_0("=" * 60)
    log_for_0(f"Checkpoint step: {loaded_step}")
    log_for_0(f"N samples: {len(hypotheses)}")
    log_for_0(f"Empty hypotheses: {empty_count}/{len(hypotheses)} ({100 * empty_count / len(hypotheses):.1f}%)")
    if ppl_results is not None:
        log_for_0(f"Gen. PPL: {ppl_results['ppl']:.4f}")
        log_for_0(f"Mean Entropy: {ppl_results['mean_entropy']:.4f}")
    else:
        log_for_0("Gen. PPL: skipped (all samples empty)")
    log_for_0("=" * 60)

    if args.out_path:
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        with open(args.out_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(hypotheses):
                f.write(json.dumps({"id": i, "generated": text}, ensure_ascii=False) + "\n")
        log_for_0(f"Saved {len(hypotheses)} generated texts to {args.out_path}")


if __name__ == "__main__":
    main()
