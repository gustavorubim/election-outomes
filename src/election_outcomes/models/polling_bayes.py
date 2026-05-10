from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta
from typing import Any, ClassVar

import numpy as np
import polars as pl

from election_outcomes.features import FeatureBundle
from election_outcomes.inference.failover import FailoverPolicy
from election_outcomes.models.common import inv_logit, logit, normal_cdf, normalize_rows
from election_outcomes.models.polling_kalman import (
    HouseEffectEstimate,
    KalmanPollingModel,
    PollObservation,
)


class BayesianPollingModel(KalmanPollingModel):
    """Opt-in Bayesian polling component with a conjugate logit-normal update.

    This is the operational Phase 1 bridge: it preserves the existing component
    schema and provenance surface while producing posterior draws and diagnostics.
    Full NumPyro NUTS can replace this fitter behind the same public methods.
    """

    component = "polling"
    POSTERIOR_SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        "draw_id": pl.Int64,
        "chain_id": pl.Int64,
        "race_id": pl.String,
        "option_id": pl.String,
        "geography": pl.String,
        "trajectory_date": pl.Date,
        "latent_logit": pl.Float64,
        "latent_share": pl.Float64,
        "systematic_error": pl.Float64,
        "pollster_effect": pl.Float64,
        "diagnostic_only": pl.Boolean,
    }

    def __init__(self, config: dict[str, object] | None = None, as_of: str | None = None) -> None:
        super().__init__(config=config, as_of=as_of)
        config = config or {}
        bayesian_config = dict(config.get("bayesian", {}))
        state_space = dict(bayesian_config.get("state_space", {}))
        self.backend = str(
            config.get("_bayesian_backend") or bayesian_config.get("backend", "analytic")
        )
        self.posterior_draw_count = int(
            bayesian_config.get("posterior_draw_count", config.get("simulation_count", 1000))
        )
        self.posterior_draw_count = max(min(self.posterior_draw_count, 5000), 100)
        self.initial_state_logit_sd = float(state_space.get("initial_state_logit_sd", 0.5))
        self.election_day_extra_sd = float(state_space.get("election_day_extra_sd", 0.025))
        self.nonsampling_logit_floor = float(
            dict(bayesian_config.get("observation", {})).get("nonsampling_logit_floor", 0.02)
        )
        self.parameterization = str(state_space.get("parameterization", "noncentered"))
        self.failover_policy = FailoverPolicy.from_config(config)
        self._config = config
        self._fallback_audit_override: dict[str, Any] | None = None
        self._fundamentals_prior = self._fundamentals_prior_lookup(
            config.get("_fundamentals_prior_rows", [])
        )
        self._cached_posterior_draws: pl.DataFrame = self._empty_posterior_draws()
        self._cached_diagnostics: dict[str, Any] = self._empty_diagnostics()

    def posterior_draws(self, bundle: FeatureBundle) -> pl.DataFrame:
        self._ensure_fit(bundle)
        return self._cached_posterior_draws

    def diagnostics(self, bundle: FeatureBundle | None = None) -> dict[str, Any]:
        if bundle is not None:
            self._ensure_fit(bundle)
        return dict(self._cached_diagnostics)

    def _fit(
        self, bundle: FeatureBundle, as_of: date | None
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame,
        dict[tuple[str, str | None], HouseEffectEstimate],
    ]:
        self._fallback_audit_override = None
        if self.backend == "nuts":
            try:
                return self._fit_nuts_backend(bundle, as_of)
            except (RuntimeError, TimeoutError, ValueError, ImportError) as exc:
                fallback_label = (
                    self.failover_policy.fallback_order[0]
                    if self.failover_policy.fallback_order
                    else "analytic_logit_normal"
                )
                self._fallback_audit_override = {
                    "fallback_used": fallback_label,
                    "failover_audit": {
                        "status": "fallback_used",
                        "primary_engine": "numpyro-nuts",
                        "fallback_used": fallback_label,
                        "reason": str(exc),
                        "timeout_seconds": self.failover_policy.timeout_seconds,
                        "fallback_order": list(self.failover_policy.fallback_order),
                        "publication_blocked": self.failover_policy.block_publication_on_fallback,
                    },
                }
                return self._fit_analytic(bundle, as_of)
        return self._fit_analytic(bundle, as_of)

    def _fit_analytic(
        self, bundle: FeatureBundle, as_of: date | None
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame,
        dict[tuple[str, str | None], HouseEffectEstimate],
    ]:
        self._cached_posterior_draws = self._empty_posterior_draws()
        self._cached_diagnostics = self._empty_diagnostics()
        if as_of is None or bundle.polls.is_empty():
            return normalize_rows([]), self._empty_trajectory(), {}

        polls = self._eligible_polls(bundle.polls, as_of)
        if polls.is_empty():
            return normalize_rows([]), self._empty_trajectory(), {}

        option_priors = self._option_priors(bundle.options)
        geography_by_race = self._geography_by_race(bundle.race_catalog)
        office_by_race = self._office_by_race(bundle.race_catalog)
        house_effects = self._estimate_house_effects(polls, option_priors)
        estimate_rows: list[dict[str, object]] = []
        trajectory_rows: list[dict[str, object]] = []
        draw_rows: list[dict[str, object]] = []
        fitted_keys: set[tuple[str, str]] = set()

        seed = self._draw_seed(bundle, as_of)
        rng = np.random.default_rng(seed)
        sort_columns = [
            column
            for column in ["race_id", "option_id", "_poll_end_date", "pollster", "poll_id"]
            if column in polls.columns
        ]
        sorted_polls = polls.sort(sort_columns) if sort_columns else polls
        posterior_sds: list[float] = []
        poll_counts: list[int] = []
        for key, group in sorted_polls.group_by(["race_id", "option_id"], maintain_order=True):
            race_id, option_id = str(key[0]), str(key[1])
            observations = [
                self._observation(row, house_effects) for row in group.iter_rows(named=True)
            ]
            observations = [observation for observation in observations if observation is not None]
            if not observations:
                continue
            fitted_keys.add((race_id, option_id))
            prior_spec = self._fundamentals_prior.get((race_id, option_id))
            if prior_spec is not None:
                prior = inv_logit(prior_spec["mean_logit"])
                prior_sd_logit = prior_spec["sd_logit"]
            else:
                prior = option_priors.get((race_id, option_id), 0.5)
                prior_sd_logit = self.initial_state_logit_sd
            mean_logit, sd_logit = self._posterior_logit(
                prior, observations, prior_sd_logit=prior_sd_logit
            )
            posterior_sds.append(sd_logit)
            poll_counts.append(len(observations))
            latent_logits = rng.normal(mean_logit, sd_logit, self.posterior_draw_count)
            latent_shares = np.array([inv_logit(float(value)) for value in latent_logits])
            vote_share = float(latent_shares.mean())
            uncertainty = float(latent_shares.std())
            marginal_win_probability = self._forecast_win_probability(mean_logit, sd_logit)
            estimate_rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "marginal_win_probability": marginal_win_probability,
                    "vote_share": vote_share,
                    "uncertainty": max(uncertainty, self.min_nonsampling_error),
                    "admitted": True,
                    "explanation": (
                        "Bayesian logit-normal polling posterior with empirical-Bayes "
                        "pollster house-effect initialization and posterior draw export."
                    ),
                }
            )
            trajectory_rows.extend(
                self._trajectory_rows_for_option(
                    race_id=race_id,
                    option_id=option_id,
                    observations=observations,
                    as_of=as_of,
                    initial_mean=prior,
                    initial_sd_logit=prior_sd_logit,
                )
            )
            geography = geography_by_race.get(race_id, "")
            mean_house_effect = self._mean_or_zero(
                [observation.house_effect for observation in observations]
            )
            draw_rows.extend(
                {
                    "draw_id": draw_id,
                    "chain_id": 0,
                    "race_id": race_id,
                    "option_id": option_id,
                    "geography": geography,
                    "trajectory_date": as_of,
                    "latent_logit": float(latent_logit),
                    "latent_share": float(latent_share),
                    "systematic_error": float(latent_logit - mean_logit),
                    "pollster_effect": mean_house_effect,
                    "diagnostic_only": False,
                }
                for draw_id, (latent_logit, latent_share) in enumerate(
                    zip(latent_logits, latent_shares, strict=True)
                )
            )

        prior_only_count = self._append_prior_only_draws(
            bundle=bundle,
            rng=rng,
            as_of=as_of,
            fitted_keys=fitted_keys,
            office_by_race=office_by_race,
            geography_by_race=geography_by_race,
            draw_rows=draw_rows,
            posterior_sds=posterior_sds,
        )
        self._cached_posterior_draws = self._posterior_frame(draw_rows)
        self._cached_diagnostics = {
            "engine": "bayes-analytic-logit-normal",
            "parameterization": self.parameterization,
            "draw_count": self.posterior_draw_count,
            "race_option_count": len(estimate_rows) + prior_only_count,
            "polling_observed_race_option_count": len(estimate_rows),
            "prior_only_race_option_count": prior_only_count,
            "poll_count": int(sum(poll_counts)),
            "fundamentals_prior_rows": len(self._fundamentals_prior),
            "fundamentals_prior_used": bool(self._fundamentals_prior),
            "posterior_logit_sd_mean": float(np.mean(posterior_sds)) if posterior_sds else None,
            "r_hat_max": None,
            "ess_min": None,
            "divergences": 0,
            "fallback_used": None,
            "failover_policy": self.failover_policy.to_dict(),
            "failover_audit": {
                "status": "not_exercised_analytic_bridge",
                "primary_engine": "bayes-analytic-logit-normal",
                "fallback_used": None,
                "publication_blocked": False,
            },
        }
        if self._fallback_audit_override:
            self._cached_diagnostics.update(self._fallback_audit_override)
        return (
            self._estimate_frame(estimate_rows),
            self._trajectory_frame(trajectory_rows),
            house_effects,
        )

    def _fit_nuts_backend(  # pragma: no cover - optional NumPyro/JAX backend
        self, bundle: FeatureBundle, as_of: date | None
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame,
        dict[tuple[str, str | None], HouseEffectEstimate],
    ]:
        if as_of is None or bundle.polls.is_empty():
            return normalize_rows([]), self._empty_trajectory(), {}
        from election_outcomes.inference.nuts import NutsConfig, fit_nuts
        from election_outcomes.inference.state_space import build_state_space_data

        eligible_polls = self._eligible_polls(bundle.polls, as_of)
        house_effects = self._estimate_house_effects(
            eligible_polls,
            self._option_priors(bundle.options),
        )
        data = build_state_space_data(
            bundle,
            as_of=as_of.isoformat(),
            office_type=None,
            prior_logit_by_key={
                key: float(value["mean_logit"]) for key, value in self._fundamentals_prior.items()
            },
            poll_half_life_days=float(
                dict(dict(self._config.get("bayesian", {})).get("state_space", {})).get(
                    "poll_half_life_days", self.half_life_days
                )
            ),
            pollster_house_effects={
                key: estimate.effect for key, estimate in house_effects.items()
            },
        )
        if data.poll_logit_y.size == 0:
            return normalize_rows([]), self._empty_trajectory(), {}
        nuts_config = dict(dict(self._config.get("bayesian", {})).get("nuts", {}))
        cfg = NutsConfig(
            num_warmup=int(nuts_config.get("num_warmup", 200)),
            num_samples=int(nuts_config.get("num_samples", self.posterior_draw_count)),
            num_chains=int(nuts_config.get("num_chains", 1)),
            chain_method=str(nuts_config.get("chain_method", "vectorized")),
            target_accept_prob=float(nuts_config.get("target_accept_prob", 0.99)),
            parameterization=self.parameterization,
            wall_clock_timeout_seconds=(
                float(nuts_config["wall_clock_timeout_seconds"])
                if nuts_config.get("wall_clock_timeout_seconds") is not None
                else None
            ),
        )
        result = fit_nuts(
            data,
            hyperpriors=self._nuts_hyperpriors(),
            config=cfg,
            seed=self._draw_seed(bundle, as_of),
        )
        state_logit = np.asarray(result.samples["state_logit"], dtype=np.float64)
        if state_logit.ndim == 1:
            state_logit = state_logit.reshape(1, -1)
        pollster_effect = np.asarray(
            result.samples.get("pollster_effect", np.zeros((state_logit.shape[0], 1))),
            dtype=np.float64,
        )
        if pollster_effect.ndim == 1:
            pollster_effect = pollster_effect.reshape(-1, 1)
        sample_count = int(state_logit.shape[0])
        if sample_count <= 0:
            raise ValueError("NUTS returned no posterior state samples")

        geography_by_race = self._geography_by_race(bundle.race_catalog)
        office_by_race = self._office_by_race(bundle.race_catalog)
        fitted_keys = set(data.race_option_keys)
        poll_counts = np.bincount(data.poll_s, minlength=len(data.race_option_keys))
        draw_rows: list[dict[str, object]] = []
        estimate_rows: list[dict[str, object]] = []
        trajectory_rows: list[dict[str, object]] = []
        posterior_sds = [float(value) for value in np.std(state_logit, axis=0)]
        for option_index, (race_id, option_id) in enumerate(data.race_option_keys):
            logits = state_logit[:, option_index]
            shares = np.array([inv_logit(float(value)) for value in logits])
            draw_logits = np.resize(logits, self.posterior_draw_count)
            draw_shares = np.resize(shares, self.posterior_draw_count)
            mean_logit = float(logits.mean())
            vote_share = float(shares.mean())
            forecast_sd_logit = self._forecast_logit_sd(float(logits.std()), vote_share)
            marginal_win_probability = self._forecast_win_probability(mean_logit, forecast_sd_logit)
            uncertainty = max(
                float(shares.std()),
                self._share_sd_from_logit_sd(vote_share, forecast_sd_logit),
                self.min_nonsampling_error,
            )
            estimate_rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "marginal_win_probability": marginal_win_probability,
                    "vote_share": vote_share,
                    "uncertainty": uncertainty,
                    "admitted": True,
                    "explanation": (
                        "Joint Bayesian state-space polling posterior fitted with NumPyro NUTS."
                    ),
                }
            )
            trajectory_rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "trajectory_date": as_of,
                    "as_of": as_of,
                    "latent_vote_share": vote_share,
                    "latent_variance": uncertainty**2,
                    "latent_sigma": uncertainty,
                    "initial_vote_share_prior": inv_logit(float(data.prior_logit[option_index])),
                    "marginal_win_probability": marginal_win_probability,
                    "poll_count": int(poll_counts[option_index]),
                    "effective_sample_size": float(result.diagnostics.get("ess_min") or 0.0),
                    "mean_observed_share": None,
                    "mean_adjusted_share": None,
                    "mean_observation_variance": None,
                    "mean_house_effect": 0.0,
                    "process_variance": 0.0,
                    "nonsampling_variance": uncertainty**2,
                    "admitted": True,
                    "explanation": "NumPyro NUTS posterior summary at forecast as-of date.",
                }
            )
            geography = geography_by_race.get(race_id, "")
            for draw_id, (latent_logit, latent_share) in enumerate(
                zip(draw_logits, draw_shares, strict=True)
            ):
                draw_rows.append(
                    {
                        "draw_id": draw_id,
                        "chain_id": 0,
                        "race_id": race_id,
                        "option_id": option_id,
                        "geography": geography,
                        "trajectory_date": as_of,
                        "latent_logit": float(latent_logit),
                        "latent_share": float(latent_share),
                        "systematic_error": float(latent_logit - mean_logit),
                        "pollster_effect": float(np.mean(pollster_effect[:, 0])),
                        "diagnostic_only": False,
                    }
                )

        rng = np.random.default_rng(self._draw_seed(bundle, as_of) + 1)
        prior_only_count = self._append_prior_only_draws(
            bundle=bundle,
            rng=rng,
            as_of=as_of,
            fitted_keys=fitted_keys,
            office_by_race=office_by_race,
            geography_by_race=geography_by_race,
            draw_rows=draw_rows,
            posterior_sds=posterior_sds,
        )
        self._cached_posterior_draws = self._posterior_frame(draw_rows)
        self._cached_diagnostics = {
            **result.diagnostics,
            "engine": "numpyro-nuts",
            "parameterization": self.parameterization,
            "draw_count": self.posterior_draw_count,
            "nuts_sample_count": sample_count,
            "race_option_count": len(data.race_option_keys) + prior_only_count,
            "polling_observed_race_option_count": len(data.race_option_keys),
            "prior_only_race_option_count": prior_only_count,
            "poll_count": int(data.poll_logit_y.size),
            "fundamentals_prior_rows": len(self._fundamentals_prior),
            "fundamentals_prior_used": bool(self._fundamentals_prior),
            "posterior_logit_sd_mean": float(np.mean(posterior_sds)) if posterior_sds else None,
            "hierarchical_effects": {
                "office_count": len(data.office_ids),
                "geography_count": len(data.geography_ids),
                "race_count": len(data.race_ids),
                "office_ids": list(data.office_ids),
                "geography_ids": list(data.geography_ids),
            },
            "fallback_used": None,
            "failover_policy": self.failover_policy.to_dict(),
        }
        return (
            self._estimate_frame(estimate_rows),
            self._trajectory_frame(trajectory_rows),
            house_effects,
        )

    def _nuts_hyperpriors(self):
        from election_outcomes.inference.state_space import HyperPriors

        bayesian = dict(self._config.get("bayesian", {}))
        state_space = dict(bayesian.get("state_space", {}))
        return HyperPriors(
            sigma_state=self.initial_state_logit_sd,
            tau_pollster=float(
                dict(bayesian.get("observation", {})).get("pollster_effect_sd", 0.04)
            ),
            sigma_office=float(
                dict(bayesian.get("cross_office", {})).get("office_offset_prior_sd", 0.02)
            ),
            sigma_geography=float(state_space.get("geography_effect_sd", 0.06)),
            sigma_race=float(state_space.get("race_effect_sd", 0.08)),
        )

    def _append_prior_only_draws(
        self,
        *,
        bundle: FeatureBundle,
        rng: np.random.Generator,
        as_of: date,
        fitted_keys: set[tuple[str, str]],
        office_by_race: dict[str, str],
        geography_by_race: dict[str, str],
        draw_rows: list[dict[str, object]],
        posterior_sds: list[float],
    ) -> int:
        candidate_offices = {"president", "senate", "house", "governor"}
        prior_only_count = 0
        for row in bundle.options.sort(["race_id", "option_id"]).iter_rows(named=True):
            race_id = str(row["race_id"])
            option_id = str(row["option_id"])
            if (race_id, option_id) in fitted_keys:
                continue
            if office_by_race.get(race_id) not in candidate_offices:
                continue
            prior_spec = self._fundamentals_prior.get((race_id, option_id))
            if prior_spec is None:
                continue
            mean_logit = float(prior_spec["mean_logit"])
            sd_logit = float(prior_spec["sd_logit"])
            posterior_sds.append(sd_logit)
            latent_logits = rng.normal(mean_logit, sd_logit, self.posterior_draw_count)
            latent_shares = np.array([inv_logit(float(value)) for value in latent_logits])
            geography = geography_by_race.get(race_id, "")
            draw_rows.extend(
                {
                    "draw_id": draw_id,
                    "chain_id": 0,
                    "race_id": race_id,
                    "option_id": option_id,
                    "geography": geography,
                    "trajectory_date": as_of,
                    "latent_logit": float(latent_logit),
                    "latent_share": float(latent_share),
                    "systematic_error": float(latent_logit - mean_logit),
                    "pollster_effect": 0.0,
                    "diagnostic_only": True,
                }
                for draw_id, (latent_logit, latent_share) in enumerate(
                    zip(latent_logits, latent_shares, strict=True)
                )
            )
            prior_only_count += 1
        return prior_only_count

    def _estimate_frame(self, rows: list[dict[str, object]]) -> pl.DataFrame:
        frame = normalize_rows(rows)
        if frame.is_empty() or "marginal_win_probability" not in frame.columns:
            return frame
        return (
            frame.with_columns(
                pl.len().over("race_id").alias("_race_option_count"),
                pl.col("marginal_win_probability")
                .sum()
                .over("race_id")
                .alias("_race_probability_sum"),
            )
            .with_columns(
                pl.when(
                    (pl.col("_race_option_count") > 1) & (pl.col("_race_probability_sum") > 0.0)
                )
                .then(pl.col("marginal_win_probability") / pl.col("_race_probability_sum"))
                .otherwise(pl.col("marginal_win_probability"))
                .alias("marginal_win_probability")
            )
            .drop(["_race_option_count", "_race_probability_sum"])
        )

    def _posterior_logit(
        self,
        prior_share: float,
        observations: list[PollObservation],
        prior_sd_logit: float | None = None,
    ) -> tuple[float, float]:
        prior_mean = logit(prior_share)
        prior_sd = self.initial_state_logit_sd if prior_sd_logit is None else prior_sd_logit
        prior_variance = max(prior_sd**2, 1e-8)
        precision = 1.0 / prior_variance
        weighted = prior_mean * precision
        for observation in observations:
            share = min(0.999999, max(0.000001, observation.adjusted_share))
            obs_logit = logit(share)
            obs_sd_share = math.sqrt(max(observation.observation_variance, 1e-10))
            obs_sd_logit = max(
                obs_sd_share / max(share * (1.0 - share), 1e-6),
                self.nonsampling_logit_floor,
            )
            obs_precision = 1.0 / max(obs_sd_logit**2, 1e-10)
            precision += obs_precision
            weighted += obs_logit * obs_precision
        posterior_mean = weighted / precision
        posterior_sd = math.sqrt((1.0 / precision) + self.election_day_extra_sd**2)
        return posterior_mean, posterior_sd

    def _forecast_logit_sd(self, posterior_sd_logit: float, mean_share: float) -> float:
        share = min(0.999999, max(0.000001, mean_share))
        floor_logit_sd = self.min_nonsampling_error / max(share * (1.0 - share), 1e-6)
        return max(
            math.sqrt(max(posterior_sd_logit, 0.0) ** 2 + self.election_day_extra_sd**2),
            floor_logit_sd,
            self.nonsampling_logit_floor,
        )

    @staticmethod
    def _share_sd_from_logit_sd(mean_share: float, logit_sd: float) -> float:
        share = min(0.999999, max(0.000001, mean_share))
        return share * (1.0 - share) * max(logit_sd, 0.0)

    def _forecast_win_probability(self, mean_logit: float, sd_logit: float) -> float:
        mean_share = inv_logit(mean_logit)
        forecast_sd_logit = self._forecast_logit_sd(sd_logit, mean_share)
        return float(normal_cdf(mean_logit / max(forecast_sd_logit, 1e-8)))

    def _trajectory_rows_for_option(
        self,
        race_id: str,
        option_id: str,
        observations: list[PollObservation],
        as_of: date,
        initial_mean: float,
        initial_sd_logit: float,
    ) -> list[dict[str, object]]:
        observations_by_date: dict[date, list[PollObservation]] = {}
        for observation in sorted(observations, key=lambda item: (item.end_date, item.poll_id)):
            observations_by_date.setdefault(observation.end_date, []).append(observation)
        if not observations_by_date:
            return []
        rows: list[dict[str, object]] = []
        start_date = min(observations_by_date)
        trajectory_date = start_date
        observed_so_far: list[PollObservation] = []
        while trajectory_date <= as_of:
            observed_so_far.extend(observations_by_date.get(trajectory_date, []))
            mean_logit, sd_logit = self._posterior_logit(
                initial_mean, observed_so_far, prior_sd_logit=initial_sd_logit
            )
            share = inv_logit(mean_logit)
            share_sd = max(share * (1.0 - share) * sd_logit, self.min_nonsampling_error)
            todays_observations = observations_by_date.get(trajectory_date, [])
            rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "trajectory_date": trajectory_date,
                    "as_of": as_of,
                    "latent_vote_share": share,
                    "latent_variance": share_sd**2,
                    "latent_sigma": share_sd,
                    "initial_vote_share_prior": initial_mean,
                    "marginal_win_probability": float(normal_cdf(mean_logit / max(sd_logit, 1e-8))),
                    "poll_count": len(todays_observations),
                    "effective_sample_size": self._mean_or_zero(
                        [observation.effective_sample_size for observation in todays_observations]
                    ),
                    "mean_observed_share": self._mean_or_none(
                        [observation.observed_share for observation in todays_observations]
                    ),
                    "mean_adjusted_share": self._mean_or_none(
                        [observation.adjusted_share for observation in todays_observations]
                    ),
                    "mean_observation_variance": self._mean_or_none(
                        [observation.observation_variance for observation in todays_observations]
                    ),
                    "mean_house_effect": self._mean_or_zero(
                        [observation.house_effect for observation in todays_observations]
                    ),
                    "process_variance": 0.0,
                    "nonsampling_variance": self.min_nonsampling_error**2,
                    "admitted": True,
                    "explanation": (
                        "Bayesian logit-normal posterior trajectory after same-day updates."
                    ),
                }
            )
            trajectory_date += timedelta(days=1)
        return rows

    def _draw_seed(self, bundle: FeatureBundle, as_of: date) -> int:
        payload = f"{self._bundle_fingerprint(bundle)}:{as_of}:{self.posterior_draw_count}:bayes"
        return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2**32)

    @staticmethod
    def _geography_by_race(race_catalog: pl.DataFrame) -> dict[str, str]:
        if race_catalog.is_empty() or not {"race_id", "geography"}.issubset(race_catalog.columns):
            return {}
        return {
            str(row["race_id"]): str(row.get("geography") or "")
            for row in race_catalog.select(["race_id", "geography"]).iter_rows(named=True)
        }

    @staticmethod
    def _office_by_race(race_catalog: pl.DataFrame) -> dict[str, str]:
        if race_catalog.is_empty() or not {"race_id", "office_type"}.issubset(race_catalog.columns):
            return {}
        return {
            str(row["race_id"]): str(row.get("office_type") or "")
            for row in race_catalog.select(["race_id", "office_type"]).iter_rows(named=True)
        }

    @classmethod
    def _posterior_frame(cls, rows: list[dict[str, object]]) -> pl.DataFrame:
        if not rows:
            return cls._empty_posterior_draws()
        return pl.DataFrame(rows, schema=cls.POSTERIOR_SCHEMA).sort(
            ["race_id", "option_id", "draw_id"]
        )

    @classmethod
    def _empty_posterior_draws(cls) -> pl.DataFrame:
        return pl.DataFrame(schema=cls.POSTERIOR_SCHEMA)

    @staticmethod
    def _fundamentals_prior_lookup(rows: object) -> dict[tuple[str, str], dict[str, float]]:
        if not isinstance(rows, list):
            return {}
        lookup: dict[tuple[str, str], dict[str, float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            race_id = row.get("race_id")
            option_id = row.get("option_id")
            mean_logit = row.get("mean_logit")
            sd_logit = row.get("sd_logit")
            if race_id is None or option_id is None or mean_logit is None or sd_logit is None:
                continue
            lookup[(str(race_id), str(option_id))] = {
                "mean_logit": float(mean_logit),
                "sd_logit": float(sd_logit),
            }
        return lookup

    def _empty_diagnostics(self) -> dict[str, Any]:
        return {
            "engine": "bayes-analytic-logit-normal",
            "draw_count": 0,
            "race_option_count": 0,
            "poll_count": 0,
            "fundamentals_prior_rows": 0,
            "fundamentals_prior_used": False,
            "r_hat_max": None,
            "ess_min": None,
            "divergences": 0,
            "fallback_used": None,
            "failover_policy": self.failover_policy.to_dict(),
            "failover_audit": {
                "status": "not_exercised_analytic_bridge",
                "primary_engine": "bayes-analytic-logit-normal",
                "fallback_used": None,
                "publication_blocked": False,
            },
        }
