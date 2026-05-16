from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import ClassVar

import polars as pl

from civic_signal.features import FeatureBundle
from civic_signal.models.common import clamp, normal_cdf, normalize_rows


@dataclass(frozen=True)
class HouseEffectEstimate:
    pollster: str
    option_id: str | None
    effect: float
    raw_effect: float
    prior_effect: float
    shrinkage: float
    poll_count: int


@dataclass(frozen=True)
class PollObservation:
    poll_id: str
    pollster: str
    end_date: date
    observed_share: float
    adjusted_share: float
    observation_variance: float
    effective_sample_size: float
    house_effect: float


class KalmanPollingModel:
    component = "polling"

    POPULATION_WEIGHTS: ClassVar[dict[str, float]] = {"lv": 1.1, "rv": 1.0, "a": 0.85}
    METHODOLOGY_WEIGHTS: ClassVar[dict[str, float]] = {
        "live_phone": 1.1,
        "mixed": 1.05,
        "online": 0.95,
    }
    TRAJECTORY_SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        "race_id": pl.String,
        "option_id": pl.String,
        "component": pl.String,
        "trajectory_date": pl.Date,
        "as_of": pl.Date,
        "latent_vote_share": pl.Float64,
        "latent_variance": pl.Float64,
        "latent_sigma": pl.Float64,
        "initial_vote_share_prior": pl.Float64,
        "marginal_win_probability": pl.Float64,
        "poll_count": pl.Int64,
        "effective_sample_size": pl.Float64,
        "mean_observed_share": pl.Float64,
        "mean_adjusted_share": pl.Float64,
        "mean_observation_variance": pl.Float64,
        "mean_house_effect": pl.Float64,
        "process_variance": pl.Float64,
        "nonsampling_variance": pl.Float64,
        "admitted": pl.Boolean,
        "explanation": pl.String,
    }

    def __init__(self, config: dict[str, object] | None = None, as_of: str | None = None) -> None:
        polling_config = dict((config or {}).get("polling", {}))
        self.half_life_days = float(polling_config.get("half_life_days", 21))
        self.min_nonsampling_error = float(polling_config.get("min_nonsampling_error", 0.035))
        self.daily_process_variance = float(polling_config.get("daily_process_variance", 0.0025**2))
        self.initial_state_variance = float(polling_config.get("initial_state_variance", 0.08**2))
        self.default_sample_size = float(polling_config.get("default_sample_size", 600))
        self.house_effect_prior_polls = float(polling_config.get("house_effect_prior_polls", 5))
        self.house_effect_iterations = max(int(polling_config.get("house_effect_iterations", 2)), 1)
        self.max_house_effect = float(polling_config.get("max_house_effect", 0.08))
        self.pollster_house_effects = {
            str(key): float(value)
            for key, value in dict(polling_config.get("pollster_house_effects", {})).items()
        }
        self.as_of = date.fromisoformat(as_of) if as_of else None
        self._cache_key: tuple[str, date | None] | None = None
        self._cached_estimates: pl.DataFrame | None = None
        self._cached_trajectory: pl.DataFrame | None = None
        self._cached_house_effects: dict[tuple[str, str | None], HouseEffectEstimate] = {}

    @property
    def cached_trajectory(self) -> pl.DataFrame | None:
        return self._cached_trajectory

    @property
    def cached_house_effects(self) -> dict[tuple[str, str | None], HouseEffectEstimate]:
        return dict(self._cached_house_effects)

    def run(self, bundle: FeatureBundle) -> pl.DataFrame:
        return self._ensure_fit(bundle)[0]

    def trajectory(self, bundle: FeatureBundle) -> pl.DataFrame:
        return self._ensure_fit(bundle)[1]

    def _ensure_fit(self, bundle: FeatureBundle) -> tuple[pl.DataFrame, pl.DataFrame]:
        as_of = self._resolve_as_of(bundle.polls)
        cache_key = (self._bundle_fingerprint(bundle), as_of)
        if (
            self._cache_key == cache_key
            and self._cached_estimates is not None
            and self._cached_trajectory is not None
        ):
            return self._cached_estimates, self._cached_trajectory

        estimates, trajectory, house_effects = self._fit(bundle, as_of)
        self._cache_key = cache_key
        self._cached_estimates = estimates
        self._cached_trajectory = trajectory
        self._cached_house_effects = house_effects
        return estimates, trajectory

    def _fit(
        self, bundle: FeatureBundle, as_of: date | None
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame,
        dict[tuple[str, str | None], HouseEffectEstimate],
    ]:
        if as_of is None or bundle.polls.is_empty():
            return normalize_rows([]), self._empty_trajectory(), {}

        polls = self._eligible_polls(bundle.polls, as_of)
        if polls.is_empty():
            return normalize_rows([]), self._empty_trajectory(), {}

        option_priors = self._option_priors(bundle.options)
        house_effects = self._estimate_house_effects(polls, option_priors)
        estimate_rows: list[dict[str, object]] = []
        trajectory_rows: list[dict[str, object]] = []

        sort_columns = [
            column
            for column in ["race_id", "option_id", "_poll_end_date", "pollster", "poll_id"]
            if column in polls.columns
        ]
        sorted_polls = polls.sort(sort_columns) if sort_columns else polls
        for key, group in sorted_polls.group_by(["race_id", "option_id"], maintain_order=True):
            race_id, option_id = key
            observations = [
                self._observation(row, house_effects) for row in group.iter_rows(named=True)
            ]
            observations = [observation for observation in observations if observation is not None]
            if not observations:
                continue
            final_state = self._fit_option_trajectory(
                str(race_id),
                str(option_id),
                observations,
                as_of,
                option_priors.get((str(race_id), str(option_id)), 0.5),
                trajectory_rows,
            )
            if final_state is None:
                continue
            share, variance = final_state
            uncertainty = max(math.sqrt(max(variance, 0.0)), self.min_nonsampling_error)
            estimate_rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "marginal_win_probability": normal_cdf((share - 0.5) / uncertainty),
                    "vote_share": share,
                    "uncertainty": uncertainty,
                    "admitted": True,
                    "explanation": (
                        "Deterministic Gaussian random-walk Kalman polling estimate "
                        "with sample-size observation variance, nonsampling floor, "
                        "and empirical-Bayes pollster house-effect shrinkage."
                    ),
                }
            )

        trajectory = self._trajectory_frame(trajectory_rows)
        return normalize_rows(estimate_rows), trajectory, house_effects

    def _fit_option_trajectory(
        self,
        race_id: str,
        option_id: str,
        observations: list[PollObservation],
        as_of: date,
        initial_mean: float,
        trajectory_rows: list[dict[str, object]],
    ) -> tuple[float, float] | None:
        observations_by_date: dict[date, list[PollObservation]] = defaultdict(list)
        for observation in sorted(observations, key=lambda item: (item.end_date, item.poll_id)):
            observations_by_date[observation.end_date].append(observation)

        start_date = min(observations_by_date)
        if start_date > as_of:
            return None

        state_mean = clamp(initial_mean, 0.001, 0.999)
        state_variance = max(self.initial_state_variance, 1e-8)
        trajectory_date = start_date
        previous_date = start_date
        final_state: tuple[float, float] | None = None
        while trajectory_date <= as_of:
            elapsed_days = max((trajectory_date - previous_date).days, 0)
            predicted_mean = state_mean
            predicted_variance = state_variance + self.daily_process_variance * elapsed_days

            todays_observations = observations_by_date.get(trajectory_date, [])
            state_mean, state_variance = self._kalman_update(
                predicted_mean, predicted_variance, todays_observations
            )
            state_mean = clamp(state_mean, 0.001, 0.999)
            state_variance = max(state_variance, 1e-10)
            final_state = (state_mean, state_variance)
            trajectory_rows.append(
                self._trajectory_row(
                    race_id,
                    option_id,
                    trajectory_date,
                    as_of,
                    state_mean,
                    state_variance,
                    initial_mean,
                    todays_observations,
                )
            )
            previous_date = trajectory_date
            trajectory_date += timedelta(days=1)
        return final_state

    def _kalman_update(
        self,
        predicted_mean: float,
        predicted_variance: float,
        observations: list[PollObservation],
    ) -> tuple[float, float]:
        state_mean = predicted_mean
        state_variance = max(predicted_variance, 1e-10)
        for observation in observations:
            observation_variance = max(observation.observation_variance, 1e-10)
            innovation_variance = state_variance + observation_variance
            kalman_gain = state_variance / innovation_variance
            state_mean += kalman_gain * (observation.adjusted_share - state_mean)
            state_variance *= 1.0 - kalman_gain
        return state_mean, state_variance

    def _trajectory_row(
        self,
        race_id: str,
        option_id: str,
        trajectory_date: date,
        as_of: date,
        state_mean: float,
        state_variance: float,
        initial_mean: float,
        observations: list[PollObservation],
    ) -> dict[str, object]:
        poll_count = len(observations)
        uncertainty = max(math.sqrt(max(state_variance, 0.0)), self.min_nonsampling_error)
        return {
            "race_id": race_id,
            "option_id": option_id,
            "component": self.component,
            "trajectory_date": trajectory_date,
            "as_of": as_of,
            "latent_vote_share": state_mean,
            "latent_variance": state_variance,
            "latent_sigma": math.sqrt(max(state_variance, 0.0)),
            "initial_vote_share_prior": initial_mean,
            "marginal_win_probability": normal_cdf((state_mean - 0.5) / uncertainty),
            "poll_count": poll_count,
            "effective_sample_size": self._mean_or_zero(
                [observation.effective_sample_size for observation in observations]
            ),
            "mean_observed_share": self._mean_or_none(
                [observation.observed_share for observation in observations]
            ),
            "mean_adjusted_share": self._mean_or_none(
                [observation.adjusted_share for observation in observations]
            ),
            "mean_observation_variance": self._mean_or_none(
                [observation.observation_variance for observation in observations]
            ),
            "mean_house_effect": self._mean_or_zero(
                [observation.house_effect for observation in observations]
            ),
            "process_variance": self.daily_process_variance,
            "nonsampling_variance": self.min_nonsampling_error**2,
            "admitted": True,
            "explanation": "Kalman daily latent polling state after same-day poll updates.",
        }

    def _observation(
        self,
        row: dict[str, object],
        house_effects: dict[tuple[str, str | None], HouseEffectEstimate],
    ) -> PollObservation | None:
        raw_end_date = row.get("_poll_end_date", row.get("end_date"))
        end_date = self._to_date(raw_end_date)
        if end_date is None:
            return None

        pollster = str(row.get("pollster") or "unknown")
        option_id = str(row.get("option_id"))
        observed_share = clamp(float(row["pct"]) / 100.0, 0.001, 0.999)
        house_effect = self._house_effect_from_estimates(pollster, option_id, house_effects)
        adjusted_share = clamp(observed_share - house_effect, 0.001, 0.999)
        effective_sample_size = self._effective_sample_size(row)
        observation_variance = self._observation_variance(observed_share, effective_sample_size)
        return PollObservation(
            poll_id=str(row.get("poll_id") or ""),
            pollster=pollster,
            end_date=end_date,
            observed_share=observed_share,
            adjusted_share=adjusted_share,
            observation_variance=observation_variance,
            effective_sample_size=effective_sample_size,
            house_effect=house_effect,
        )

    def _estimate_house_effects(
        self,
        polls: pl.DataFrame,
        option_priors: dict[tuple[str, str], float],
    ) -> dict[tuple[str, str | None], HouseEffectEstimate]:
        estimates = self._static_house_effects(polls)
        for _iteration in range(self.house_effect_iterations - 1):
            estimates = self._trajectory_house_effects(polls, option_priors, estimates)
        return estimates

    def _static_house_effects(
        self, polls: pl.DataFrame
    ) -> dict[tuple[str, str | None], HouseEffectEstimate]:
        rows = list(polls.iter_rows(named=True))
        if not rows:
            return {}

        group_totals: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
        prepared: list[tuple[dict[str, object], float, float]] = []
        for row in rows:
            share = clamp(float(row["pct"]) / 100.0, 0.001, 0.999)
            effective_sample_size = self._effective_sample_size(row)
            variance = self._observation_variance(share, effective_sample_size)
            weight = 1.0 / max(variance, 1e-10)
            key = (str(row["race_id"]), str(row["option_id"]))
            group_totals[key][0] += weight * share
            group_totals[key][1] += weight
            prepared.append((row, share, weight))

        means = {
            key: weighted_total / weight_total
            for key, (weighted_total, weight_total) in group_totals.items()
            if weight_total > 0
        }
        residual_groups: dict[tuple[str, str | None], list[tuple[float, float]]] = defaultdict(list)
        for row, share, weight in prepared:
            pollster = str(row.get("pollster") or "unknown")
            option_id = str(row["option_id"])
            mean = means.get((str(row["race_id"]), option_id))
            if mean is None:
                continue
            residual = share - mean
            residual_groups[(pollster, option_id)].append((residual, weight))
            residual_groups[(pollster, None)].append((residual, weight))

        estimates: dict[tuple[str, str | None], HouseEffectEstimate] = {}
        for (pollster, option_id), residuals in residual_groups.items():
            weight_total = sum(weight for _residual, weight in residuals)
            if weight_total <= 0:
                continue
            raw_effect = sum(residual * weight for residual, weight in residuals) / weight_total
            prior_effect = self._configured_house_effect(pollster, option_id)
            poll_count = len(residuals)
            shrinkage = poll_count / (poll_count + max(self.house_effect_prior_polls, 0.0))
            effect = prior_effect + shrinkage * (raw_effect - prior_effect)
            estimates[(pollster, option_id)] = HouseEffectEstimate(
                pollster=pollster,
                option_id=option_id,
                effect=clamp(effect, -self.max_house_effect, self.max_house_effect),
                raw_effect=raw_effect,
                prior_effect=prior_effect,
                shrinkage=shrinkage,
                poll_count=poll_count,
            )
        return estimates

    def _trajectory_house_effects(
        self,
        polls: pl.DataFrame,
        option_priors: dict[tuple[str, str], float],
        current: dict[tuple[str, str | None], HouseEffectEstimate],
    ) -> dict[tuple[str, str | None], HouseEffectEstimate]:
        residual_groups: dict[tuple[str, str | None], list[tuple[float, float]]] = defaultdict(list)
        sort_columns = [
            column
            for column in ["race_id", "option_id", "_poll_end_date", "pollster", "poll_id"]
            if column in polls.columns
        ]
        sorted_polls = polls.sort(sort_columns) if sort_columns else polls
        for key, group in sorted_polls.group_by(["race_id", "option_id"], maintain_order=True):
            race_id, option_id = str(key[0]), str(key[1])
            observations = [self._observation(row, current) for row in group.iter_rows(named=True)]
            observations = [observation for observation in observations if observation is not None]
            if not observations:
                continue
            state_by_date = self._filtered_states_by_date(
                observations,
                option_priors.get((race_id, option_id), 0.5),
            )
            for observation in observations:
                reference_share = state_by_date.get(observation.end_date)
                if reference_share is None:
                    continue
                residual = observation.observed_share - reference_share
                weight = 1.0 / max(observation.observation_variance, 1e-10)
                residual_groups[(observation.pollster, option_id)].append((residual, weight))
                residual_groups[(observation.pollster, None)].append((residual, weight))
        return self._shrink_house_effect_groups(residual_groups)

    def _filtered_states_by_date(
        self, observations: list[PollObservation], initial_mean: float
    ) -> dict[date, float]:
        observations_by_date: dict[date, list[PollObservation]] = defaultdict(list)
        for observation in sorted(observations, key=lambda item: (item.end_date, item.poll_id)):
            observations_by_date[observation.end_date].append(observation)
        state_mean = clamp(initial_mean, 0.001, 0.999)
        state_variance = max(self.initial_state_variance, 1e-8)
        previous_date = min(observations_by_date)
        state_by_date: dict[date, float] = {}
        for observation_date in sorted(observations_by_date):
            elapsed_days = max((observation_date - previous_date).days, 0)
            predicted_variance = state_variance + self.daily_process_variance * elapsed_days
            state_mean, state_variance = self._kalman_update(
                state_mean,
                predicted_variance,
                observations_by_date[observation_date],
            )
            state_mean = clamp(state_mean, 0.001, 0.999)
            state_variance = max(state_variance, 1e-10)
            state_by_date[observation_date] = state_mean
            previous_date = observation_date
        return state_by_date

    def _shrink_house_effect_groups(
        self, residual_groups: dict[tuple[str, str | None], list[tuple[float, float]]]
    ) -> dict[tuple[str, str | None], HouseEffectEstimate]:
        estimates: dict[tuple[str, str | None], HouseEffectEstimate] = {}
        for (pollster, option_id), residuals in residual_groups.items():
            weight_total = sum(weight for _residual, weight in residuals)
            if weight_total <= 0:
                continue
            raw_effect = sum(residual * weight for residual, weight in residuals) / weight_total
            prior_effect = self._configured_house_effect(pollster, option_id)
            poll_count = len(residuals)
            shrinkage = poll_count / (poll_count + max(self.house_effect_prior_polls, 0.0))
            effect = prior_effect + shrinkage * (raw_effect - prior_effect)
            estimates[(pollster, option_id)] = HouseEffectEstimate(
                pollster=pollster,
                option_id=option_id,
                effect=clamp(effect, -self.max_house_effect, self.max_house_effect),
                raw_effect=raw_effect,
                prior_effect=prior_effect,
                shrinkage=shrinkage,
                poll_count=poll_count,
            )
        return estimates

    def _house_effect_from_estimates(
        self,
        pollster: str,
        option_id: str,
        estimates: dict[tuple[str, str | None], HouseEffectEstimate],
    ) -> float:
        estimate = estimates.get((pollster, option_id)) or estimates.get((pollster, None))
        if estimate is not None:
            return estimate.effect
        return self._configured_house_effect(pollster, option_id)

    def _configured_house_effect(self, pollster: str, option_id: str | None) -> float:
        option_key = f"{pollster}:{option_id}" if option_id is not None else ""
        return self.pollster_house_effects.get(
            option_key,
            self.pollster_house_effects.get(pollster, 0.0),
        )

    def _effective_sample_size(self, row: dict[str, object]) -> float:
        sample = max(float(row.get("sample_size") or self.default_sample_size), 1.0)
        population_weight = self.POPULATION_WEIGHTS.get(str(row.get("population")), 1.0)
        methodology_weight = self.METHODOLOGY_WEIGHTS.get(str(row.get("methodology")), 1.0)
        sponsor_weight = 1.0 if str(row.get("sponsor_class")) == "nonpartisan" else 0.85
        return max(sample * population_weight * methodology_weight * sponsor_weight, 1.0)

    def _observation_variance(self, share: float, effective_sample_size: float) -> float:
        sampling_variance = max(share * (1.0 - share), 1e-6) / max(effective_sample_size, 1.0)
        return sampling_variance + self.min_nonsampling_error**2

    def _eligible_polls(self, polls: pl.DataFrame, as_of: date) -> pl.DataFrame:
        if polls.is_empty() or "end_date" not in polls.columns:
            return polls.clear()
        date_expr = self._date_expr("end_date")
        return (
            polls.with_columns(date_expr.alias("_poll_end_date"))
            .filter(pl.col("_poll_end_date").is_not_null() & (pl.col("_poll_end_date") <= as_of))
            .sort(
                [
                    column
                    for column in ["race_id", "option_id", "_poll_end_date", "pollster", "poll_id"]
                    if column in polls.columns or column == "_poll_end_date"
                ]
            )
        )

    def _resolve_as_of(self, polls: pl.DataFrame) -> date | None:
        if self.as_of is not None:
            return self.as_of
        if polls.is_empty() or "end_date" not in polls.columns:
            return None
        max_value = polls.select(self._date_expr("end_date").max()).item()
        return self._to_date(max_value)

    @staticmethod
    def _option_priors(options: pl.DataFrame) -> dict[tuple[str, str], float]:
        if options.is_empty() or not {"race_id", "option_id", "previous_vote_share"}.issubset(
            set(options.columns)
        ):
            return {}
        priors: dict[tuple[str, str], float] = {}
        for row in options.select(["race_id", "option_id", "previous_vote_share"]).iter_rows(
            named=True
        ):
            value = row.get("previous_vote_share")
            if value is None:
                continue
            priors[(str(row["race_id"]), str(row["option_id"]))] = clamp(float(value), 0.001, 0.999)
        return priors

    def _bundle_fingerprint(self, bundle: FeatureBundle) -> str:
        payload = {
            "polls": self._frame_fingerprint(bundle.polls),
            "options": self._frame_fingerprint(bundle.options),
            "settings": {
                "daily_process_variance": self.daily_process_variance,
                "default_sample_size": self.default_sample_size,
                "half_life_days": self.half_life_days,
                "house_effect_iterations": self.house_effect_iterations,
                "house_effect_prior_polls": self.house_effect_prior_polls,
                "initial_state_variance": self.initial_state_variance,
                "max_house_effect": self.max_house_effect,
                "min_nonsampling_error": self.min_nonsampling_error,
                "pollster_house_effects": self.pollster_house_effects,
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _frame_fingerprint(frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "empty"
        ordered = frame.select(sorted(frame.columns))
        if ordered.columns:
            ordered = ordered.sort(ordered.columns)
        payload = json.dumps(ordered.to_dicts(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _date_expr(column: str) -> pl.Expr:
        if column == "_poll_end_date":
            return pl.col(column)
        text = pl.col(column).cast(pl.Utf8)
        return pl.coalesce(
            pl.col(column).cast(pl.Date, strict=False),
            text.str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        )

    @staticmethod
    def _to_date(value: object) -> date | None:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])

    @classmethod
    def _trajectory_frame(cls, rows: list[dict[str, object]]) -> pl.DataFrame:
        if not rows:
            return cls._empty_trajectory()
        return pl.DataFrame(rows, schema=cls.TRAJECTORY_SCHEMA).sort(
            ["race_id", "option_id", "trajectory_date"]
        )

    @classmethod
    def _empty_trajectory(cls) -> pl.DataFrame:
        return pl.DataFrame(schema=cls.TRAJECTORY_SCHEMA)

    @staticmethod
    def _mean_or_none(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _mean_or_zero(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)
