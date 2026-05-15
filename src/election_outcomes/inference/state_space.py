from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from election_outcomes.features import FeatureBundle, filter_bundle_by_date, subset_bundle
from election_outcomes.models.common import logit


@dataclass(frozen=True)
class StateSpaceData:
    poll_t: np.ndarray
    poll_s: np.ndarray
    poll_j: np.ndarray
    poll_o: np.ndarray
    poll_logit_y: np.ndarray
    poll_kappa: np.ndarray
    prior_logit: np.ndarray
    option_office: np.ndarray
    option_geography: np.ndarray
    option_race: np.ndarray
    race_option_keys: list[tuple[str, str]]
    pollster_ids: list[str]
    office_ids: list[str]
    geography_ids: list[str]
    race_ids: list[str]
    dims: tuple[int, int, int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HyperPriors:
    sigma_state: float = 0.35
    tau_pollster: float = 0.04
    sigma_office: float = 0.08
    sigma_geography: float = 0.06
    sigma_race: float = 0.08


def build_state_space_data(
    bundle: FeatureBundle,
    *,
    as_of: str,
    office_type: str | None = "president",
    cycle: int | None = None,
    prior_logit_by_key: dict[tuple[str, str], float] | None = None,
    poll_half_life_days: float = 21.0,
    process_drift_sd_per_sqrt_day: float = 0.0,
    pollster_house_effects: dict[tuple[str, str | None], float] | None = None,
) -> StateSpaceData:
    """Build poll-level tensors consumed by the hierarchical NumPyro backend."""

    cutoff = date.fromisoformat(as_of)
    catalog = bundle.race_catalog
    if office_type is not None and "office_type" in catalog.columns:
        catalog = catalog.filter(pl.col("office_type") == office_type)
    if cycle is not None and "cycle" in catalog.columns:
        catalog = catalog.filter(pl.col("cycle") == cycle)
    active = filter_bundle_by_date(subset_bundle(bundle, catalog), as_of)
    if active.polls.is_empty():
        return _empty_state_space_data({"as_of": as_of, "office_type": office_type, "cycle": cycle})

    poll_date_expr = pl.coalesce(["end_date", "start_date"]).alias("_poll_date")
    polls = (
        active.polls.with_columns(poll_date_expr)
        .filter(pl.col("_poll_date").is_not_null() & (pl.col("_poll_date") <= cutoff))
        .filter(pl.col("pct").is_not_null())
        .sort(["race_id", "option_id", "_poll_date", "pollster", "poll_id"])
    )
    if polls.is_empty():
        return _empty_state_space_data({"as_of": as_of, "office_type": office_type, "cycle": cycle})

    option_prior = _option_prior(active.options)
    race_metadata = _race_metadata(active.race_catalog)
    keys = sorted(
        {
            (str(row["race_id"]), str(row["option_id"]))
            for row in polls.select(["race_id", "option_id"]).iter_rows(named=True)
        }
    )
    race_ids = sorted({race_id for race_id, _option_id in keys})
    office_ids = sorted(
        {race_metadata.get(race_id, {}).get("office", "unknown") for race_id in race_ids}
    )
    geography_ids = sorted(
        {race_metadata.get(race_id, {}).get("geography_group", "unknown") for race_id in race_ids}
    )
    pollsters = sorted(str(value) for value in polls["pollster"].fill_null("unknown").unique())
    key_index = {key: index for index, key in enumerate(keys)}
    pollster_index = {pollster: index for index, pollster in enumerate(pollsters)}
    race_index = {race_id: index for index, race_id in enumerate(race_ids)}
    office_index = {office_id: index for index, office_id in enumerate(office_ids)}
    geography_index = {geography_id: index for index, geography_id in enumerate(geography_ids)}
    min_date = polls["_poll_date"].min()
    if not hasattr(min_date, "toordinal"):
        min_date = date.fromisoformat(str(min_date))

    poll_t: list[int] = []
    poll_s: list[int] = []
    poll_j: list[int] = []
    poll_o: list[int] = []
    y: list[float] = []
    kappa: list[float] = []
    observation_weights: list[float] = []
    house_effect_values: list[float] = []
    house_effect_lookup = pollster_house_effects or {}
    for row in polls.iter_rows(named=True):
        key = (str(row["race_id"]), str(row["option_id"]))
        pollster = str(row.get("pollster") or "unknown")
        option_id = str(row["option_id"])
        poll_date = row["_poll_date"]
        if not hasattr(poll_date, "toordinal"):
            poll_date = date.fromisoformat(str(poll_date))
        observed_share = min(0.999, max(0.001, float(row["pct"]) / 100.0))
        house_effect = house_effect_lookup.get(
            (pollster, option_id),
            house_effect_lookup.get((pollster, None), 0.0),
        )
        share = min(0.999, max(0.001, observed_share - house_effect))
        sample_size = max(float(row.get("sample_size") or 600.0), 50.0)
        quality_weight = _poll_quality_weight(row)
        age_days = max((cutoff - poll_date).days, 0)
        recency_weight = _recency_weight(age_days, poll_half_life_days)
        observation_weight = max(quality_weight * recency_weight, 1e-3)
        effective_sample_size = sample_size * observation_weight
        share_sd = math.sqrt(max(share * (1.0 - share) / effective_sample_size, 1e-6))
        obs_sd_logit = share_sd / max(share * (1.0 - share), 1e-6)
        process_sd_logit = max(float(process_drift_sd_per_sqrt_day), 0.0) * math.sqrt(age_days)
        poll_t.append(int((poll_date - min_date).days))
        poll_s.append(key_index[key])
        poll_j.append(pollster_index[pollster])
        poll_o.append(0)
        y.append(logit(share))
        kappa.append(max(math.sqrt(obs_sd_logit**2 + process_sd_logit**2), 0.02))
        observation_weights.append(observation_weight)
        house_effect_values.append(float(house_effect))

    prior_lookup = prior_logit_by_key or {}
    prior_logit = np.array(
        [
            float(prior_lookup[key]) if key in prior_lookup else logit(option_prior.get(key, 0.5))
            for key in keys
        ],
        dtype=np.float64,
    )
    option_office = np.array(
        [
            office_index[race_metadata.get(race_id, {}).get("office", "unknown")]
            for race_id, _option_id in keys
        ],
        dtype=np.int64,
    )
    option_geography = np.array(
        [
            geography_index[race_metadata.get(race_id, {}).get("geography_group", "unknown")]
            for race_id, _option_id in keys
        ],
        dtype=np.int64,
    )
    option_race = np.array(
        [race_index[race_id] for race_id, _option_id in keys],
        dtype=np.int64,
    )
    return StateSpaceData(
        poll_t=np.array(poll_t, dtype=np.int64),
        poll_s=np.array(poll_s, dtype=np.int64),
        poll_j=np.array(poll_j, dtype=np.int64),
        poll_o=np.array(poll_o, dtype=np.int64),
        poll_logit_y=np.array(y, dtype=np.float64),
        poll_kappa=np.array(kappa, dtype=np.float64),
        prior_logit=prior_logit,
        option_office=option_office,
        option_geography=option_geography,
        option_race=option_race,
        race_option_keys=keys,
        pollster_ids=pollsters,
        office_ids=office_ids,
        geography_ids=geography_ids,
        race_ids=race_ids,
        dims=(len(keys), int(max(poll_t, default=0) + 1), len(pollsters)),
        metadata={
            "as_of": as_of,
            "office_type": office_type,
            "cycle": cycle,
            "poll_count": len(y),
            "poll_half_life_days": float(poll_half_life_days),
            "process_drift_sd_per_sqrt_day": float(process_drift_sd_per_sqrt_day),
            "temporal_process_variance": "poll_age_logit_variance",
            "observation_weight_min": float(min(observation_weights, default=0.0)),
            "observation_weight_max": float(max(observation_weights, default=0.0)),
            "observation_weight_mean": float(np.mean(observation_weights))
            if observation_weights
            else 0.0,
            "pollster_house_effect_adjustment_mean_abs": float(np.mean(np.abs(house_effect_values)))
            if house_effect_values
            else 0.0,
            "race_option_count": len(keys),
            "pollster_count": len(pollsters),
            "office_count": len(office_ids),
            "geography_count": len(geography_ids),
            "race_count": len(race_ids),
            "hierarchy": {
                "office_ids": office_ids,
                "geography_ids": geography_ids,
                "race_ids": race_ids,
            },
        },
    )


def state_space_model(
    data: StateSpaceData,
    hyperpriors: HyperPriors | None = None,
    *,
    parameterization: str = "noncentered",
) -> None:  # pragma: no cover
    """NumPyro hierarchical polling model used by the opt-in NUTS backend."""

    try:
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
    except ImportError as exc:
        raise RuntimeError("NumPyro/JAX are required; run `uv sync`.") from exc

    priors = hyperpriors or HyperPriors()
    state_count, _time_count, pollster_count = data.dims
    office_count = len(data.office_ids)
    geography_count = len(data.geography_ids)
    race_count = len(data.race_ids)
    sigma_state = numpyro.sample("sigma_state", dist.HalfNormal(priors.sigma_state))
    sigma_office = numpyro.sample("sigma_office", dist.HalfNormal(priors.sigma_office))
    sigma_geography = numpyro.sample("sigma_geography", dist.HalfNormal(priors.sigma_geography))
    sigma_race = numpyro.sample("sigma_race", dist.HalfNormal(priors.sigma_race))
    tau_pollster = numpyro.sample("tau_pollster", dist.HalfNormal(priors.tau_pollster))
    prior = jnp.asarray(data.prior_logit)
    option_office = jnp.asarray(data.option_office)
    option_geography = jnp.asarray(data.option_geography)
    option_race = jnp.asarray(data.option_race)
    if parameterization == "noncentered":
        office_z = numpyro.sample("office_z", dist.Normal(0.0, 1.0).expand([max(office_count, 1)]))
        geography_z = numpyro.sample(
            "geography_z", dist.Normal(0.0, 1.0).expand([max(geography_count, 1)])
        )
        race_z = numpyro.sample("race_z", dist.Normal(0.0, 1.0).expand([max(race_count, 1)]))
        state_z = numpyro.sample("state_z", dist.Normal(0.0, 1.0).expand([state_count]))
        office_effect = numpyro.deterministic(
            "office_effect", _centered_effect(sigma_office * office_z)
        )
        geography_effect = numpyro.deterministic(
            "geography_effect", _centered_effect(sigma_geography * geography_z)
        )
        race_effect = numpyro.deterministic("race_effect", _centered_effect(sigma_race * race_z))
        option_effect = sigma_state * state_z
        state_logit = numpyro.deterministic(
            "state_logit",
            prior
            + office_effect[option_office]
            + geography_effect[option_geography]
            + race_effect[option_race]
            + option_effect,
        )
    elif parameterization == "centered":
        office_effect = numpyro.deterministic(
            "office_effect",
            _centered_effect(
                numpyro.sample(
                    "office_effect_raw",
                    dist.Normal(0.0, sigma_office).expand([max(office_count, 1)]),
                )
            ),
        )
        geography_effect = numpyro.deterministic(
            "geography_effect",
            _centered_effect(
                numpyro.sample(
                    "geography_effect_raw",
                    dist.Normal(0.0, sigma_geography).expand([max(geography_count, 1)]),
                )
            ),
        )
        race_effect = numpyro.deterministic(
            "race_effect",
            _centered_effect(
                numpyro.sample(
                    "race_effect_raw",
                    dist.Normal(0.0, sigma_race).expand([max(race_count, 1)]),
                )
            ),
        )
        state_logit = numpyro.sample(
            "state_logit",
            dist.Normal(
                prior
                + office_effect[option_office]
                + geography_effect[option_geography]
                + race_effect[option_race],
                sigma_state,
            ),
        )
    else:
        raise ValueError("parameterization must be 'centered' or 'noncentered'")
    raw_pollster = numpyro.sample(
        "pollster_raw", dist.Normal(0.0, 1.0).expand([max(pollster_count, 1)])
    )
    pollster_effect = numpyro.deterministic(
        "pollster_effect",
        tau_pollster * (raw_pollster - jnp.mean(raw_pollster)),
    )
    mu = state_logit[jnp.asarray(data.poll_s)] + pollster_effect[jnp.asarray(data.poll_j)]
    numpyro.sample(
        "poll_logit_y",
        dist.Normal(mu, jnp.asarray(data.poll_kappa)),
        obs=jnp.asarray(data.poll_logit_y),
    )


def _centered_effect(values: Any) -> Any:  # pragma: no cover
    return values - values.mean()


def _option_prior(options: pl.DataFrame) -> dict[tuple[str, str], float]:
    if options.is_empty():
        return {}
    priors: dict[tuple[str, str], float] = {}
    for row in options.iter_rows(named=True):
        value = row.get("previous_vote_share")
        priors[(str(row["race_id"]), str(row["option_id"]))] = (
            float(value) if value is not None else 0.5
        )
    return priors


def _race_metadata(races: pl.DataFrame) -> dict[str, dict[str, str]]:
    if races.is_empty():
        return {}
    metadata: dict[str, dict[str, str]] = {}
    required = {"race_id", "office_type", "geography"}
    columns = [
        column for column in ["race_id", "office_type", "geography"] if column in races.columns
    ]
    if not required.issubset(set(columns)):
        return {}
    for row in races.select(columns).iter_rows(named=True):
        race_id = str(row["race_id"])
        geography = str(row.get("geography") or "unknown")
        metadata[race_id] = {
            "office": str(row.get("office_type") or "unknown").lower(),
            "geography_group": _geography_group(geography),
        }
    return metadata


def _geography_group(geography: str) -> str:
    if not geography:
        return "unknown"
    if "-" in geography:
        return geography.split("-", 1)[0]
    return geography


def _poll_quality_weight(row: dict[str, Any]) -> float:
    population = str(row.get("population") or row.get("population_full") or "").lower()
    methodology = str(row.get("methodology") or "").lower()
    population_weight = {"lv": 1.1, "likely": 1.1, "rv": 1.0, "registered": 1.0, "a": 0.85}
    methodology_weight = {
        "live_phone": 1.1,
        "mixed": 1.05,
        "online": 0.95,
        "ivr": 0.9,
        "text": 0.9,
    }
    pop_weight = next(
        (weight for key, weight in population_weight.items() if key in population),
        1.0,
    )
    method_weight = next(
        (weight for key, weight in methodology_weight.items() if key in methodology),
        1.0,
    )
    return max(pop_weight * method_weight, 0.1)


def _recency_weight(age_days: int, half_life_days: float) -> float:
    half_life = max(float(half_life_days), 1.0)
    return 0.5 ** (max(age_days, 0) / half_life)


def _empty_state_space_data(metadata: dict[str, Any]) -> StateSpaceData:
    return StateSpaceData(
        poll_t=np.array([], dtype=np.int64),
        poll_s=np.array([], dtype=np.int64),
        poll_j=np.array([], dtype=np.int64),
        poll_o=np.array([], dtype=np.int64),
        poll_logit_y=np.array([], dtype=np.float64),
        poll_kappa=np.array([], dtype=np.float64),
        prior_logit=np.array([], dtype=np.float64),
        option_office=np.array([], dtype=np.int64),
        option_geography=np.array([], dtype=np.int64),
        option_race=np.array([], dtype=np.int64),
        race_option_keys=[],
        pollster_ids=[],
        office_ids=[],
        geography_ids=[],
        race_ids=[],
        dims=(0, 0, 0),
        metadata=metadata
        | {
            "poll_count": 0,
            "race_option_count": 0,
            "pollster_count": 0,
            "office_count": 0,
            "geography_count": 0,
            "race_count": 0,
            "hierarchy": {"office_ids": [], "geography_ids": [], "race_ids": []},
        },
    )
