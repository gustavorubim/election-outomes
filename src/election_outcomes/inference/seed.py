from __future__ import annotations

import hashlib


def derive_seed(model_config_hash: str, run_id: str, salt: str = "phase0") -> int:
    """Derive a stable 32-bit seed from immutable run inputs."""

    payload = f"{model_config_hash}:{run_id}:{salt}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2**32)


def jax_prng_key(seed: int):
    try:
        import jax

        return jax.random.PRNGKey(int(seed))
    except ImportError as exc:
        raise RuntimeError("JAX is required for PRNG keys; run `uv sync`.") from exc
