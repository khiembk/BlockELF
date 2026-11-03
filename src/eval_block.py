#!/usr/bin/env python
"""Evaluate a block-revelation checkpoint with block-to-block (Algorithm 2) generation.

Unlike eval.py (which uses the standard one-shot full-sequence sampler), this drives
generation through `utils.generation_utils.generate_block_cond` — the same fixed-b
committed-context revelation loop the model was trained for — then scores BLEU/ROUGE
against the references. Single-device, eager Python loop (no pmap): intended for
checking in on a checkpoint's quality, not high-throughput batch eval.
"""
import argparse
import copy
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
from utils.data_utils import load_dataset_split, get_dataloader, get_pad_token_id
from utils.generation_utils import generate_block_cond
from utils.metrics_utils import compute_bleu, compute_rouge
from configs.config import load_config_from_yaml, apply_config_overrides, load_sampling_configs

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ELF block-revelation checkpoint via block-to-block generation")
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
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)

    log_for_0(f"Loading eval dataset from {config.eval_data_path}...")
    eval_dataset = load_dataset_split(config.eval_data_path)
    log_for_0(f"Eval dataset size: {len(eval_dataset)}")

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

    dataloader = get_dataloader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    hypotheses, references, sources, printed = [], [], [], 0
    samples_done = 0
    for batch in dataloader:
        if samples_done >= args.num_samples:
            break
        source_ids = jnp.array(batch["input_ids"])[:, :config.max_input_length]
        source_lengths = jnp.array(batch["cond_seq_mask"]).astype(jnp.int32).sum(axis=1)

        rng, gen_rng = jax.random.split(rng)
        texts = generate_block_cond(
            state=state, encoder_params=encoder_params, encoder_apply_fn=encoder_model.apply,
            tokenizer=tokenizer, config=config, sampling_config=sampling_config,
            source_ids=source_ids, source_lengths=source_lengths, rng=gen_rng,
        )
        batch_n = len(texts)
        refs = batch["target"]
        srcs = batch["input"]
        hypotheses.extend(texts)
        references.extend(refs)
        sources.extend(srcs)
        samples_done += batch_n

        while printed < args.print_examples and printed < len(hypotheses):
            log_for_0(f"[{printed}] SRC: {sources[printed]}")
            log_for_0(f"    REF: {references[printed]}")
            log_for_0(f"    HYP: {hypotheses[printed]}")
            printed += 1

        log_for_0(f"Generated {samples_done}/{args.num_samples}...")

    bleu = compute_bleu(hypotheses, references)
    rouge = compute_rouge(hypotheses, references)
    empty_count = sum(1 for h in hypotheses if not h.strip())
    log_for_0("=" * 60)
    log_for_0(f"Checkpoint step: {loaded_step}")
    log_for_0(f"N samples: {len(hypotheses)}")
    log_for_0(f"Empty hypotheses: {empty_count}/{len(hypotheses)} ({100 * empty_count / len(hypotheses):.1f}%)")
    log_for_0(f"BLEU: {bleu:.2f}")
    log_for_0(f"ROUGE-1/2/L: {rouge['rouge1']:.2f} / {rouge['rouge2']:.2f} / {rouge['rougeL']:.2f}")
    log_for_0("=" * 60)


if __name__ == "__main__":
    main()
