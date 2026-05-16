from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from civic_signal.features import FeatureBundle
from civic_signal.models.fundamentals import FundamentalsModel


@dataclass(frozen=True)
class FundamentalsPrior:
    race_ids: list[str]
    option_ids: list[str]
    mean_logit: np.ndarray
    sd_logit: np.ndarray
    prior_method: list[str]
    structural_sd: float
    prior_strength: float
    frame: pl.DataFrame


@dataclass(frozen=True)
class DiagonalNormalPrior:
    mean: np.ndarray
    sd: np.ndarray


def build_fundamentals_prior(
    fundamentals_model: FundamentalsModel,
    bundle: FeatureBundle,
    config: dict[str, Any],
) -> FundamentalsPrior:
    bayesian_cfg = dict(config.get("bayesian", {}))
    prior_cfg = dict(bayesian_cfg.get("fundamentals_prior", {}))
    prior_strength = max(float(prior_cfg.get("prior_strength", 0.5)), 1e-6)
    frame = fundamentals_model.predictive_distribution(bundle)
    if frame.is_empty():
        return FundamentalsPrior(
            race_ids=[],
            option_ids=[],
            mean_logit=np.array([], dtype=np.float64),
            sd_logit=np.array([], dtype=np.float64),
            prior_method=[],
            structural_sd=float(prior_cfg.get("structural_sd", 0.05)),
            prior_strength=prior_strength,
            frame=frame,
        )
    frame = frame.with_columns(
        (pl.col("sd_logit") / np.sqrt(prior_strength)).alias("sd_logit"),
        pl.lit(prior_strength).alias("prior_strength"),
    )
    ordered = frame.sort(["race_id", "option_id"])
    return FundamentalsPrior(
        race_ids=[str(value) for value in ordered["race_id"].to_list()],
        option_ids=[str(value) for value in ordered["option_id"].to_list()],
        mean_logit=ordered["mean_logit"].to_numpy().astype(np.float64),
        sd_logit=ordered["sd_logit"].to_numpy().astype(np.float64),
        prior_method=[str(value) for value in ordered["prior_method"].to_list()],
        structural_sd=float(ordered["structural_sd_logit"].max()),
        prior_strength=prior_strength,
        frame=ordered,
    )


def to_numpyro_prior(
    fp: FundamentalsPrior,
    state_index: dict[str, int],
) -> Any:
    mean = np.zeros(len(state_index), dtype=np.float64)
    sd = np.ones(len(state_index), dtype=np.float64)
    for race_id, value, scale in zip(fp.race_ids, fp.mean_logit, fp.sd_logit, strict=True):
        if race_id in state_index:
            index = state_index[race_id]
        else:
            state_key = race_id.split("-")[-2] if "-" in race_id else race_id
            if state_key not in state_index:
                continue
            index = state_index[state_key]
        mean[index] = float(value)
        sd[index] = float(scale)
    try:
        import jax.numpy as jnp
        import numpyro.distributions as dist

        return dist.Normal(jnp.asarray(mean), jnp.asarray(sd))
    except ImportError:
        return DiagonalNormalPrior(mean=mean, sd=sd)
