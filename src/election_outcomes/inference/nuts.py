from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from election_outcomes.inference.failover import FailoverPolicy, execute_with_failover
from election_outcomes.inference.seed import jax_prng_key
from election_outcomes.inference.state_space import HyperPriors, StateSpaceData, state_space_model


@dataclass(frozen=True)
class NutsConfig:
    num_warmup: int = 500
    num_samples: int = 2000
    num_chains: int = 2
    chain_method: str = "vectorized"
    target_accept_prob: float = 0.99
    parameterization: str = "noncentered"
    wall_clock_timeout_seconds: float | None = None


@dataclass(frozen=True)
class InferenceResult:
    samples: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
    elapsed_seconds: float

    def posterior_mean(self, name: str) -> np.ndarray:
        values = self.samples[name]
        return np.asarray(values).mean(axis=0)


def fit_nuts(
    data: StateSpaceData,
    hyperpriors: HyperPriors | None = None,
    config: NutsConfig | None = None,
    seed: int = 0,
) -> InferenceResult:  # pragma: no cover
    if data.poll_logit_y.size == 0:
        raise ValueError("StateSpaceData contains no poll observations")
    try:
        import numpyro
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise RuntimeError("NumPyro/JAX are required; run `uv sync`.") from exc

    numpyro.enable_x64()
    cfg = config or NutsConfig()
    kernel = NUTS(
        lambda: state_space_model(
            data,
            hyperpriors or HyperPriors(),
            parameterization=cfg.parameterization,
        ),
        target_accept_prob=cfg.target_accept_prob,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
        chain_method=cfg.chain_method,
        progress_bar=False,
    )
    started = time.perf_counter()
    timeout_result = execute_with_failover(
        lambda: mcmc.run(jax_prng_key(seed)),
        fallback=None,
        policy=FailoverPolicy(timeout_seconds=cfg.wall_clock_timeout_seconds),
        primary_engine="numpyro-nuts",
    )
    elapsed = time.perf_counter() - started
    samples = {key: np.asarray(value) for key, value in mcmc.get_samples().items()}
    grouped_samples = {
        key: np.asarray(value) for key, value in mcmc.get_samples(group_by_chain=True).items()
    }
    diagnostics = _diagnostics(mcmc, cfg, data, elapsed, grouped_samples)
    diagnostics["failover_audit"] = timeout_result.audit
    return InferenceResult(samples=samples, diagnostics=diagnostics, elapsed_seconds=elapsed)


def _diagnostics(  # pragma: no cover
    mcmc: Any,
    config: NutsConfig,
    data: StateSpaceData,
    elapsed: float,
    grouped_samples: dict[str, np.ndarray],
) -> dict[str, Any]:
    extra_fields = mcmc.get_extra_fields()
    divergences = int(np.asarray(extra_fields.get("diverging", np.array([]))).sum())
    quality = _quality_metrics(grouped_samples, config.num_chains)
    return {
        "engine": "numpyro-nuts",
        "num_warmup": config.num_warmup,
        "num_samples": config.num_samples,
        "num_chains": config.num_chains,
        "chain_method": config.chain_method,
        "target_accept_prob": config.target_accept_prob,
        "parameterization": config.parameterization,
        "elapsed_seconds": elapsed,
        "divergences": divergences,
        "r_hat_max": quality["r_hat_max"],
        "ess_min": quality["ess_min"],
        "r_hat_available": quality["r_hat_available"],
        "ess_available": quality["ess_available"],
        "hierarchy": {
            "office_count": len(data.office_ids),
            "geography_count": len(data.geography_ids),
            "race_count": len(data.race_ids),
            "office_ids": list(data.office_ids),
            "geography_ids": list(data.geography_ids),
        },
    }


def _quality_metrics(  # pragma: no cover
    grouped_samples: dict[str, np.ndarray], num_chains: int
) -> dict[str, Any]:
    try:
        from numpyro.diagnostics import summary
    except ImportError:
        return {
            "r_hat_max": None,
            "ess_min": None,
            "r_hat_available": False,
            "ess_available": False,
        }

    diagnostics = summary(grouped_samples, group_by_chain=True)
    r_hat_values: list[float] = []
    ess_values: list[float] = []
    for parameter in diagnostics.values():
        n_eff = parameter.get("n_eff")
        if n_eff is not None:
            ess_values.extend(_finite_values(n_eff))
        r_hat = parameter.get("r_hat")
        if r_hat is not None and num_chains >= 2:
            r_hat_values.extend(_finite_values(r_hat))
    return {
        "r_hat_max": max(r_hat_values) if r_hat_values else None,
        "ess_min": min(ess_values) if ess_values else None,
        "r_hat_available": bool(r_hat_values),
        "ess_available": bool(ess_values),
    }


def _finite_values(value: Any) -> list[float]:  # pragma: no cover
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(item) for item in values if np.isfinite(item)]
