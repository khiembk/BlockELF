import jax
import jax.numpy as jnp
import flax.linen as nn

from modules.layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder,
    DEFAULT_KERNEL_INIT, DEFAULT_BIAS_INIT, NORMAL_INIT_002,
)


class ELFBlock(nn.Module):
    """ELF Transformer block."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, rope_fn=None, attention_mask=None, deterministic=True):
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)

        x_normed = RMSNorm(self.hidden_size, eps=1e-6, name='norm1')(x)
        attn_out = Attention(
            self.hidden_size, self.num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=self.attn_drop, proj_drop=self.proj_drop, name='attn',
        )(x_normed, rope_fn, attention_mask=attention_mask, deterministic=deterministic)
        x = x + attn_out

        x_normed = RMSNorm(self.hidden_size, eps=1e-6, name='norm2')(x)
        mlp_out = SwiGLUFFN(self.hidden_size, mlp_hidden_dim, drop=self.proj_drop, name='mlp')(
            x_normed, deterministic=deterministic,
        )
        x = x + mlp_out
        return x


class ELF(nn.Module):
    """Text ELF Transformer."""
    text_encoder_dim: int
    max_length: int
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    bottleneck_dim: int = 128
    num_time_tokens: int = 4  # Number of in-context time conditioning tokens
    num_self_cond_cfg_tokens: int = 4  # Number of in-context self-cond CFG tokens
    num_model_mode_tokens: int = 0  # If > 0, prepend learnable model-mode tokens that signal decoding mode
    vocab_size: int = 0  # Vocabulary size for decoder unembedding

    def build_context(self, t, self_cond_cfg_scale=None):
        prefix_tokens = []
        B = t.shape[0]

        def _make_prefix(emb, n_tokens, param_name):
            tokens = self.param(param_name, NORMAL_INIT_002, (1, n_tokens, self.hidden_size))
            return jnp.tile(tokens, (B, 1, 1)) + jnp.expand_dims(emb, 1)

        if self.num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")
        time_emb = TimestepEmbedder(self.hidden_size, name='t_embedder')(t)
        prefix_tokens.append(_make_prefix(time_emb, self.num_time_tokens, 't_emb_tokens'))

        if self_cond_cfg_scale is not None:
            sc_emb = TimestepEmbedder(self.hidden_size, name='self_cond_cfg_embedder')(self_cond_cfg_scale)
            if self.num_self_cond_cfg_tokens > 0:
                prefix_tokens.append(_make_prefix(sc_emb, self.num_self_cond_cfg_tokens, 'self_cond_cfg_tokens'))

        return prefix_tokens

    @nn.compact
    def __call__(
        self, x, t, attention_mask=None, deterministic=True,
        self_cond_cfg_scale=None, decoder_step_active=None, t_pos=None, role_clean_mask=None,
    ):
        """x: (N, S, C) or (N, S, 2C) with self-cond. t: (N,). attention_mask: (N, S), 1=valid.

        t_pos: optional (N, S) per-position effective timestep, added directly into the
        content tokens (on top of the existing global `t` prefix-token conditioning) for
        staggered-noise training (Adaptive BlockFlow Stage 1 + SetDLM-style single-pass
        multi-round curriculum — see utils/block_utils.compute_staggered_timesteps).
        None (default) for every existing non-staggered config/checkpoint: in that case
        no extra params are created here and behavior is exactly as before.

        role_clean_mask: optional (N, S) bool, 1 at "clean" content positions (source +
        already-committed target, during block-revelation training/inference) — Fix C:
        builds a role-based self-attention mask so clean queries can only attend to
        clean keys (their representation must stay invariant to active/noisy content),
        while active queries still attend to everything (clean + active), matching
        `utils.block_utils.build_committed_context`'s rule for the frozen T5 encoder,
        now applied to this model's own self-attention too. Prepended special tokens
        (prefix time/self-cond-cfg tokens, model-mode tokens) are treated as "clean"
        queries (so their own representation can't be indirectly tainted by active
        content either) but remain visible as *keys* to every query, clean or active,
        so active positions still receive time/CFG/mode conditioning. None (default):
        no behavior change, `attention_mask` is used exactly as before (unrestricted
        bidirectional attention when also None, as in every pre-Fix-C config).
        """
        patch_size = 1
        head_dim = self.hidden_size // self.num_heads
        B = x.shape[0]

        # Self-conditioning: input is [z, x_pred] when 2x encoder dim
        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = nn.Dense(
                self.text_encoder_dim, use_bias=True,
                kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT, name='self_cond_proj',
            )(x)

        # Text projection (with bottleneck)
        x = BottleneckTextProj(
            self.text_encoder_dim, self.hidden_size, self.bottleneck_dim, name='text_proj',
        )(x)

        if t_pos is not None:
            pos_time_emb = TimestepEmbedder(self.hidden_size, name='pos_t_embedder')(t_pos)
            x = x + pos_time_emb

        # Prepend learnable model-mode tokens (gated: zero unless decoder_step_active=True)
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = jnp.tile(
                self.param('mode_tokens', NORMAL_INIT_002,
                           (1, self.num_model_mode_tokens, self.hidden_size)),
                (B, 1, 1),
            )
            active_gate = jnp.array(False) if decoder_step_active is None else decoder_step_active
            mode_tokens = mode_tokens * active_gate.astype(mode_tokens.dtype)
            x = jnp.concatenate([mode_tokens, x], axis=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = jnp.ones((B, self.num_model_mode_tokens), dtype=attention_mask.dtype)
                attention_mask = jnp.concatenate([mode_mask, attention_mask], axis=1)

        prefix_len = 0
        context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)
        if context_prefix_tokens:
            prefix_tokens = jnp.concatenate(context_prefix_tokens, axis=1)
            prefix_len = prefix_tokens.shape[1]
            x = jnp.concatenate([prefix_tokens, x], axis=1)
            if attention_mask is not None:
                prefix_mask = jnp.ones((B, prefix_len), dtype=attention_mask.dtype)
                attention_mask = jnp.concatenate([prefix_mask, attention_mask], axis=1)

        if role_clean_mask is not None:
            # Fix C: build the full (B, S_total, S_total) role-based query/key mask,
            # covering the prepended special tokens too (treated as "clean" queries,
            # but visible as keys to every query -- see docstring). Overrides whatever
            # the generic `attention_mask is not None` extension above would have done,
            # since that logic only ever chains a simple 1-D validity mask, not a role.
            is_clean_special = jnp.ones((B, model_mode_offset + prefix_len), dtype=bool)
            is_clean_full = jnp.concatenate([is_clean_special, role_clean_mask.astype(bool)], axis=1)
            is_clean_q = is_clean_full[:, :, None]
            is_clean_k = is_clean_full[:, None, :]
            attention_mask = jnp.where(is_clean_q, is_clean_k, jnp.ones_like(is_clean_k)).astype(x.dtype)

        feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=self.max_length,
            num_empty_token=prefix_len + model_mode_offset, name='feat_rope',
        )

        q1, q3 = self.depth // 4, self.depth // 4 * 3
        for i in range(self.depth):
            in_drop_range = q3 > i >= q1
            block = ELFBlock(
                self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio,
                attn_drop=self.attn_drop if in_drop_range else 0.0,
                proj_drop=self.proj_drop if in_drop_range else 0.0,
                name=f'blocks_{i}',
            )
            x = block(x, rope_fn=feat_rope, attention_mask=attention_mask, deterministic=deterministic)

        x = x[:, prefix_len + model_mode_offset:]

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab
        decoder_logits = None
        bn = self.text_encoder_dim
        proj_kernel = self.param('proj_kernel', DEFAULT_KERNEL_INIT, (self.hidden_size, bn))
        proj_bias = self.param('proj_bias', DEFAULT_BIAS_INIT, (bn,))
        unembed_kernel = self.param('unembed_kernel', DEFAULT_KERNEL_INIT, (bn, self.vocab_size))
        unembed_bias = self.param('unembed_bias', DEFAULT_BIAS_INIT, (self.vocab_size,))
        if decoder_step_active is not None:
            decoder_logits = jax.lax.cond(
                decoder_step_active,
                lambda xi: jax.nn.gelu(xi @ proj_kernel + proj_bias) @ unembed_kernel + unembed_bias,
                lambda xi: jnp.zeros((*xi.shape[:2], self.vocab_size), dtype=xi.dtype),
                x,
            )

        output = FinalLayer(self.hidden_size, patch_size, self.text_encoder_dim, name='final_layer')(x)
        return output, decoder_logits


# Model factory functions
def ELF_B(**kwargs): return ELF(depth=12, hidden_size=768,  num_heads=12, **kwargs)
def ELF_M(**kwargs): return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)
def ELF_L(**kwargs): return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)

ELF_models = {
    'ELF-B': ELF_B, 'ELF-M': ELF_M, 'ELF-L': ELF_L,
}
