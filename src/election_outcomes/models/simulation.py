from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import polars as pl

from election_outcomes.features import FeatureBundle
from election_outcomes.performance.kernels import (
    NUMBA_AVAILABLE,
    configure_numba_threads,
    simulate_binary_draw_arrays,
)


@dataclass(frozen=True)
class SimulationOutputs:
    draws: pl.DataFrame
    race_forecasts: pl.DataFrame
    control_forecasts: pl.DataFrame
    ecosystem_forecasts: pl.DataFrame
    performance: dict[str, object]


class SimulationEngine:
    def __init__(
        self,
        config: dict[str, object],
        residual_covariance: pl.DataFrame | None = None,
        holdovers: dict[str, int] | None = None,
    ) -> None:
        self.config = config
        self.residual_covariance = residual_covariance
        self.holdovers: dict[str, int] = {
            str(key).upper(): int(value) for key, value in (holdovers or {}).items()
        }
        self.seed = int(config.get("seed", 20260508))
        self.draw_count = int(config.get("simulation_count", 1000))
        uncertainty = dict(config.get("uncertainty", {}))
        self.tier_sigma = {
            "A": float(uncertainty.get("tier_a_sigma", 0.035)),
            "B": float(uncertainty.get("tier_b_sigma", 0.075)),
        }
        self.heavy_tail_df = max(float(uncertainty.get("heavy_tail_df", 5)), 2.5)
        self.heavy_tail_scale = float(np.sqrt(self.heavy_tail_df / (self.heavy_tail_df - 2.0)))
        correlation = dict(config.get("correlation", {}))
        self.national_sigma = float(correlation.get("national_sigma", 0.015))
        self.region_sigma = float(correlation.get("region_sigma", 0.010))
        self.office_sigma = float(correlation.get("office_sigma", 0.008))
        self.geographic_groups = {
            str(key): str(value)
            for key, value in dict(correlation.get("geographic_groups", {})).items()
        }
        self.control_thresholds = {
            str(key): int(value)
            for key, value in dict(config.get("control_thresholds", {})).items()
        }
        performance = dict(config.get("performance", {}))
        requested_engine = str(performance.get("engine", "numba"))
        self.parallel = bool(performance.get("parallel", True))
        self.numba_threads = configure_numba_threads(int(performance.get("numba_threads", 0) or 0))
        self.use_numba = requested_engine == "numba" and self.parallel and NUMBA_AVAILABLE
        self.engine = "numba" if self.use_numba else "python"
        self.requested_engine = requested_engine

    def run(self, bundle: FeatureBundle, ensemble: pl.DataFrame) -> SimulationOutputs:
        draws = self._draws(bundle, ensemble)
        forecasts = self._race_forecasts(bundle, draws, ensemble)
        control = self._control_forecasts(bundle, draws)
        ecosystem = self._ecosystem_forecasts(bundle, draws)
        return SimulationOutputs(draws, forecasts, control, ecosystem, self.performance_metadata())

    def performance_metadata(self) -> dict[str, object]:
        return {
            "requested_engine": self.requested_engine,
            "engine": self.engine,
            "parallel": self.parallel,
            "numba_available": NUMBA_AVAILABLE,
            "numba_threads": self.numba_threads,
            "simulation_count": self.draw_count,
        }

    def _draws(self, bundle: FeatureBundle, ensemble: pl.DataFrame) -> pl.DataFrame:
        if ensemble.is_empty():
            return pl.DataFrame()
        rng = np.random.default_rng(self.seed)
        catalog = {row["race_id"]: row for row in bundle.race_catalog.iter_rows(named=True)}
        options_by_race = {}
        for key, group in bundle.options.group_by("race_id", maintain_order=True):
            race_key = key[0] if isinstance(key, tuple) else key
            options_by_race[str(race_key)] = group.sort("option_id")
        estimates = {
            row["race_id"]: row
            for row in ensemble.sort("option_id")
            .group_by("race_id", maintain_order=True)
            .map_groups(lambda group: group.head(1))
            .iter_rows(named=True)
        }
        fundamentals = {row["race_id"]: row for row in bundle.fundamentals.iter_rows(named=True)}
        binary_specs: list[dict[str, object]] = []
        multi_option_specs: list[dict[str, object]] = []
        national_error = rng.normal(0, self.national_sigma, self.draw_count)
        systematic_errors = self._systematic_errors(catalog, rng)
        for race_id, options in options_by_race.items():
            race = catalog[race_id]
            if race["tier"] == "C" or race_id not in estimates:
                continue
            estimate_rows = ensemble.filter(pl.col("race_id") == race_id).sort("option_id")
            first = estimate_rows.row(0, named=True)
            sigma = max(
                self.tier_sigma.get(str(race["tier"]), 0.08), float(first["uncertainty"]) * 0.5
            )
            if len(options) == 2:
                binary_specs.append(
                    {
                        "race_id": race_id,
                        "options": options,
                        "first_share": float(first["vote_share"]),
                        "turnout_base": self._turnout_base(str(race_id), fundamentals),
                        "local_error": systematic_errors[race_id]
                        + rng.standard_t(df=self.heavy_tail_df, size=self.draw_count)
                        * sigma
                        / self.heavy_tail_scale,
                    }
                )
                continue
            multi_option_specs.append(
                {
                    "race_id": race_id,
                    "options": options,
                    "estimate_rows": estimate_rows,
                    "turnout_base": self._turnout_base(str(race_id), fundamentals),
                    "systematic_error": systematic_errors[race_id],
                    "local_error": rng.standard_t(df=self.heavy_tail_df, size=self.draw_count)
                    * sigma
                    / self.heavy_tail_scale,
                }
            )

        frames: list[pl.DataFrame] = []
        binary_frame = self._binary_draw_frame(binary_specs, national_error)
        if not binary_frame.is_empty():
            frames.append(binary_frame)
        multi_frame = self._multi_option_draw_frame(multi_option_specs, national_error, rng)
        if not multi_frame.is_empty():
            frames.append(multi_frame)
        return pl.concat(frames, how="vertical") if frames else pl.DataFrame()

    def _binary_draw_frame(
        self, specs: list[dict[str, object]], national_error: np.ndarray
    ) -> pl.DataFrame:
        if not specs:
            return pl.DataFrame()
        first_shares = np.array([spec["first_share"] for spec in specs], dtype=np.float64)
        turnout_bases = np.array([spec["turnout_base"] for spec in specs], dtype=np.float64)
        local_errors = np.vstack([spec["local_error"] for spec in specs]).astype(np.float64)
        (
            draw_ids,
            correlated_error_draw_ids,
            race_indices,
            option_indices,
            turnouts,
            vote_shares,
            winners,
        ) = simulate_binary_draw_arrays(
            first_shares,
            turnout_bases,
            national_error.astype(np.float64),
            local_errors,
            self.use_numba,
        )
        draw_frame = pl.DataFrame(
            {
                "draw_id": draw_ids,
                "correlated_error_draw_id": correlated_error_draw_ids,
                "race_index": race_indices,
                "option_index": option_indices,
                "turnout": turnouts,
                "vote_share": vote_shares,
                "winner": winners,
            }
        )
        race_map = pl.DataFrame(
            {
                "race_index": list(range(len(specs))),
                "race_id": [str(spec["race_id"]) for spec in specs],
            }
        )
        option_rows = []
        for race_index, spec in enumerate(specs):
            options = spec["options"]
            for option_index, option in enumerate(options.iter_rows(named=True)):
                option_rows.append(
                    {
                        "race_index": race_index,
                        "option_index": option_index,
                        "option_id": option["option_id"],
                        "party": option["party"],
                    }
                )
        option_map = pl.DataFrame(option_rows)
        return (
            draw_frame.join(race_map, on="race_index", how="left")
            .join(option_map, on=["race_index", "option_index"], how="left")
            .select(
                [
                    "draw_id",
                    "correlated_error_draw_id",
                    "race_id",
                    "option_id",
                    "party",
                    "turnout",
                    "vote_share",
                    "winner",
                ]
            )
        )

    def _multi_option_draw_frame(
        self,
        specs: list[dict[str, object]],
        national_error: np.ndarray,
        rng: np.random.Generator,
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for spec in specs:
            race_id = str(spec["race_id"])
            options = spec["options"]
            option_shares = self._multi_option_shares(spec["estimate_rows"], rng)
            turnout_base = float(spec["turnout_base"])
            systematic_error = spec["systematic_error"]
            local_error = spec["local_error"]
            for draw_id in range(self.draw_count):
                shares = self._apply_multi_option_error(
                    [float(series[draw_id]) for series in option_shares],
                    float(
                        national_error[draw_id] + systematic_error[draw_id] + local_error[draw_id]
                    ),
                )
                winner_index = int(np.argmax(shares))
                turnout = round(turnout_base * max(0.6, 1 + national_error[draw_id]))
                for index, option in enumerate(options.iter_rows(named=True)):
                    rows.append(
                        {
                            "draw_id": draw_id,
                            "correlated_error_draw_id": draw_id,
                            "race_id": race_id,
                            "option_id": option["option_id"],
                            "party": option["party"],
                            "turnout": turnout,
                            "vote_share": shares[index],
                            "winner": index == winner_index,
                        }
                    )
        return pl.DataFrame(rows)

    def _systematic_errors(
        self,
        catalog: dict[str, dict[str, object]],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        if self.residual_covariance is not None and not self.residual_covariance.is_empty():
            return self._covariance_systematic_errors(catalog, rng)
        regions = sorted(
            {self._region_for_race(row) for row in catalog.values() if row.get("tier") != "C"}
        )
        offices = sorted(
            {str(row.get("office_type")) for row in catalog.values() if row.get("tier") != "C"}
        )
        region_errors = {
            region: rng.normal(0, self.region_sigma, self.draw_count) for region in regions
        }
        office_errors = {
            office: rng.normal(0, self.office_sigma, self.draw_count) for office in offices
        }
        return {
            race_id: region_errors[self._region_for_race(row)]
            + office_errors[str(row.get("office_type"))]
            for race_id, row in catalog.items()
            if row.get("tier") != "C"
        }

    def _covariance_systematic_errors(
        self,
        catalog: dict[str, dict[str, object]],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        active = {race_id: row for race_id, row in catalog.items() if row.get("tier") != "C"}
        groups = sorted({self._covariance_group_for_race(row) for row in active.values()})
        regions = sorted({self._region_for_race(row) for row in active.values()})
        offices = sorted({str(row.get("office_type")) for row in active.values()})
        covariance_lookup = {
            (str(row["row_group"]), str(row["column_group"])): float(row["covariance"])
            for row in self.residual_covariance.iter_rows(named=True)
        }
        fallback_variance = max(self.region_sigma**2, 0.0004)
        matrix = np.zeros((len(groups), len(groups)), dtype=np.float64)
        for row_index, row_group in enumerate(groups):
            for column_index, column_group in enumerate(groups):
                matrix[row_index, column_index] = covariance_lookup.get(
                    (row_group, column_group),
                    fallback_variance if row_index == column_index else 0.0,
                )
        matrix = (matrix + matrix.T) / 2.0
        matrix += np.eye(len(groups)) * 1e-8
        group_draws = rng.multivariate_normal(
            mean=np.zeros(len(groups)), cov=matrix, size=self.draw_count
        ).T
        group_errors = {group: group_draws[index] for index, group in enumerate(groups)}
        region_errors = {
            region: rng.normal(0, self.region_sigma, self.draw_count) for region in regions
        }
        office_errors = {
            office: rng.normal(0, self.office_sigma, self.draw_count) for office in offices
        }
        return {
            race_id: group_errors[self._covariance_group_for_race(row)]
            + region_errors[self._region_for_race(row)]
            + office_errors[str(row.get("office_type"))]
            for race_id, row in active.items()
        }

    def _region_for_race(self, race: dict[str, object]) -> str:
        geography = str(race.get("geography") or "")
        state = geography.split("-")[0]
        return self.geographic_groups.get(state, state or "unknown")

    @staticmethod
    def _covariance_group_for_race(race: dict[str, object]) -> str:
        geography = str(race.get("geography") or "")
        return geography.split("-")[0] or "unknown"

    @staticmethod
    def _apply_multi_option_error(shares: list[float], error: float) -> list[float]:
        if len(shares) <= 1:
            return list(shares)
        arr = np.clip(np.array(shares, dtype=np.float64), 1e-6, None)
        log_shares = np.log(arr)
        centered = log_shares - log_shares.mean()
        spread = max(float(np.std(centered)), 1e-3)
        perturbed = log_shares + error * (centered / spread)
        perturbed = perturbed - perturbed.max()
        new_shares = np.exp(perturbed)
        new_shares = new_shares / new_shares.sum()
        new_shares = np.clip(new_shares, 0.02, 0.98)
        return (new_shares / new_shares.sum()).tolist()

    def _multi_option_shares(
        self,
        estimate_rows: pl.DataFrame,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        shares = estimate_rows.sort("option_id")["vote_share"].to_numpy()
        alpha = np.maximum(shares * 70, 1.0)
        sampled = rng.dirichlet(alpha, size=self.draw_count)
        return [sampled[:, index] for index in range(sampled.shape[1])]

    @staticmethod
    def _turnout_base(race_id: str, fundamentals: dict[str, dict[str, object]]) -> float:
        row = fundamentals.get(race_id, {})
        voters = float(row.get("registered_voters") or 100_000)
        turnout_rate = float(row.get("historical_turnout_rate") or 0.5)
        return voters * turnout_rate

    def _race_forecasts(
        self, bundle: FeatureBundle, draws: pl.DataFrame, ensemble: pl.DataFrame
    ) -> pl.DataFrame:
        catalog = bundle.race_catalog
        options = bundle.options
        driver_columns = [
            "race_id",
            "option_id",
            "explanation",
            "component_contributions",
            "uncertainty",
        ]
        drivers = (
            ensemble.select([column for column in driver_columns if column in ensemble.columns])
            if not ensemble.is_empty()
            else pl.DataFrame()
        )
        if not drivers.is_empty():
            drivers = drivers.rename(
                {"explanation": "top_drivers", "uncertainty": "model_uncertainty"}
            )
        if draws.is_empty():
            base = options.join(
                catalog.select(["race_id", "tier", "tier_reason"]), on="race_id", how="left"
            )
            empty = base.select(
                "race_id",
                "option_id",
                "tier",
                "tier_reason",
                pl.lit(None, dtype=pl.Float64).alias("winner_probability"),
            )
            return self._attach_forecast_explainability(empty, drivers)
        intervals = draws.group_by(["race_id", "option_id"]).agg(
            pl.col("winner").mean().alias("winner_probability"),
            pl.col("vote_share").mean().alias("vote_share_mean"),
            pl.col("vote_share").median().alias("vote_share_median"),
            pl.col("vote_share").quantile(0.25).alias("vote_share_p25"),
            pl.col("vote_share").quantile(0.75).alias("vote_share_p75"),
            pl.col("vote_share").quantile(0.10).alias("vote_share_p10"),
            pl.col("vote_share").quantile(0.90).alias("vote_share_p90"),
            pl.col("vote_share").quantile(0.05).alias("vote_share_p05"),
            pl.col("vote_share").quantile(0.95).alias("vote_share_p95"),
            pl.col("vote_share").quantile(0.025).alias("vote_share_p025"),
            pl.col("vote_share").quantile(0.975).alias("vote_share_p975"),
        )
        base = options.join(
            catalog.select(["race_id", "tier", "tier_reason"]), on="race_id", how="left"
        )
        joined = base.join(intervals, on=["race_id", "option_id"], how="left")
        forecast = joined.with_columns(
            pl.when(pl.col("tier") == "C")
            .then(None)
            .otherwise(pl.col("winner_probability"))
            .alias("winner_probability"),
            pl.when(pl.col("tier") == "C")
            .then(pl.lit("probability_withheld"))
            .otherwise(pl.lit("trusted_probability"))
            .alias("data_quality_flags"),
        )
        return self._attach_forecast_explainability(forecast, drivers)

    @staticmethod
    def _attach_forecast_explainability(
        forecast: pl.DataFrame, drivers: pl.DataFrame
    ) -> pl.DataFrame:
        if not drivers.is_empty():
            forecast = forecast.join(drivers, on=["race_id", "option_id"], how="left")
        for column in ("top_drivers", "component_contributions", "model_uncertainty"):
            if column not in forecast.columns:
                forecast = forecast.with_columns(pl.lit(None).alias(column))
        return forecast.with_columns(
            pl.when(pl.col("tier") == "C")
            .then(pl.lit("probability withheld by tier gate"))
            .otherwise(pl.col("top_drivers").fill_null("no admitted component"))
            .alias("top_drivers"),
            pl.when(pl.col("tier") == "C")
            .then(pl.lit("{}"))
            .otherwise(pl.col("component_contributions").fill_null("{}"))
            .alias("component_contributions"),
            pl.when(pl.col("tier") == "C")
            .then(pl.lit("Tier C has insufficient validated data for probability output."))
            .otherwise(
                pl.concat_str(
                    [
                        pl.lit("Simulation uncertainty combines component posterior proxy, "),
                        pl.lit("tier floor, heavy-tailed local residuals, and systematic factors."),
                    ]
                )
            )
            .alias("uncertainty_explanation"),
        )

    def _control_forecasts(self, bundle: FeatureBundle, draws: pl.DataFrame) -> pl.DataFrame:
        if draws.is_empty():
            return pl.DataFrame()
        catalog = bundle.race_catalog.select(["race_id", "office_type", "control_body", "seats"])
        joined = draws.join(
            catalog,
            on="race_id",
            how="left",
        ).filter(pl.col("control_body").is_not_null())
        winner_draws = joined.filter(pl.col("winner"))
        rows: list[dict[str, object]] = []
        for key, _group in joined.group_by(["control_body", "party"], maintain_order=True):
            control_body, party = key
            if not control_body:
                continue
            party_winners = winner_draws.filter(
                (pl.col("control_body") == control_body) & (pl.col("party") == party)
            )
            counts_by_draw = (
                party_winners.group_by("draw_id")
                .agg(pl.col("seats").sum().alias("seat_count"))
                .sort("draw_id")
            )
            count_map = {
                row["draw_id"]: float(row["seat_count"])
                for row in counts_by_draw.iter_rows(named=True)
            }
            modeled_counts = np.array(
                [count_map.get(draw_id, 0.0) for draw_id in range(self.draw_count)]
            )
            modeled_seats = self._modeled_seats(joined, str(control_body))
            threshold = self._control_threshold(str(control_body), modeled_seats)
            holdover_seats = int(self.holdovers.get(str(party).upper(), 0))
            counts = modeled_counts + holdover_seats
            tipping = self._pivotal_races(
                joined=joined,
                counts=counts,
                control_body=str(control_body),
                party=str(party),
                threshold=threshold,
            )
            seats_to_majority = max(threshold - int(np.mean(counts)), 0)
            rows.append(
                {
                    "control_body": control_body,
                    "party": party,
                    "control_threshold": threshold,
                    "modeled_seats": modeled_seats,
                    "holdover_seats": holdover_seats,
                    "control_scope": "configured_threshold"
                    if str(control_body) in self.control_thresholds
                    else "modeled_races_majority",
                    "seat_count_mean": float(np.mean(counts)),
                    "seat_count_modeled_mean": float(np.mean(modeled_counts)),
                    "seat_count_p10": float(np.quantile(counts, 0.10)),
                    "seat_count_p50": float(np.quantile(counts, 0.50)),
                    "seat_count_p90": float(np.quantile(counts, 0.90)),
                    "majority_probability": float(np.mean(counts >= threshold)),
                    "control_probability": float(np.mean(counts >= threshold)),
                    "seats_to_majority_mean": seats_to_majority,
                    "tipping_point_races": json.dumps(
                        [item["race_id"] for item in tipping[:3]], sort_keys=True
                    ),
                    "pivotal_rates": json.dumps(tipping[:3], sort_keys=True),
                }
            )
        return pl.DataFrame(rows)

    def _control_threshold(self, control_body: str, modeled_seats: int) -> int:
        return self.control_thresholds.get(control_body, max(modeled_seats // 2 + 1, 1))

    @staticmethod
    def _modeled_seats(joined: pl.DataFrame, control_body: str) -> int:
        return int(
            joined.filter(pl.col("control_body") == control_body)
            .select(["race_id", "seats"])
            .unique()
            .select(pl.col("seats").sum())
            .item()
            or 0
        )

    def _pivotal_races(
        self,
        joined: pl.DataFrame,
        counts: np.ndarray,
        control_body: str,
        party: str,
        threshold: int,
    ) -> list[dict[str, object]]:
        rows = []
        body_party = joined.filter(
            (pl.col("control_body") == control_body) & (pl.col("party") == party)
        )
        for race_id, group in body_party.group_by("race_id", maintain_order=True):
            race_key = str(race_id[0] if isinstance(race_id, tuple) else race_id)
            seat_count = int(group.select(pl.col("seats").max()).item() or 1)
            wins = {
                row["draw_id"]: bool(row["winner"])
                for row in group.select(["draw_id", "winner"]).iter_rows(named=True)
            }
            pivotal = 0
            for draw_id in range(self.draw_count):
                party_won_race = wins.get(draw_id, False)
                count = counts[draw_id]
                if party_won_race and count >= threshold and count - seat_count < threshold:
                    pivotal += 1
                elif (not party_won_race) and count < threshold and count + seat_count >= threshold:
                    pivotal += 1
            rows.append({"race_id": race_key, "pivotal_rate": pivotal / self.draw_count})
        return sorted(rows, key=lambda item: item["pivotal_rate"], reverse=True)

    def _ecosystem_forecasts(self, bundle: FeatureBundle, draws: pl.DataFrame) -> pl.DataFrame:
        catalog = {row["race_id"]: row for row in bundle.race_catalog.iter_rows(named=True)}
        rows: list[dict[str, object]] = []
        for race_id, group in draws.group_by("race_id", maintain_order=True):
            race_key = str(race_id[0] if isinstance(race_id, tuple) else race_id)
            pivot = group.group_by("draw_id").agg(
                pl.col("vote_share").sort(descending=True).head(2).alias("top_two"),
                pl.col("turnout").max().alias("turnout"),
            )
            margins = np.array([values[0] - values[1] for values in pivot["top_two"].to_list()])
            turnout = pivot["turnout"].to_numpy()
            rows.append(
                {
                    "race_id": race_key,
                    "tier": catalog[race_key]["tier"],
                    "turnout_mean": float(np.mean(turnout)),
                    "turnout_p10": float(np.quantile(turnout, 0.10)),
                    "turnout_p90": float(np.quantile(turnout, 0.90)),
                    "demographic_composition": json.dumps(
                        {
                            "status": "placeholder",
                            "supported": False,
                            "reason": "No group-level turnout model is implemented yet.",
                        },
                        sort_keys=True,
                    ),
                    "demographic_model_status": "placeholder_not_estimated",
                    "recount_probability": float(np.mean(margins <= 0.01)),
                    "certification_risk_probability": float(np.mean(margins <= 0.005) * 0.6),
                    "certification_risk_model": "close_margin_proxy_not_calibrated",
                    "ballot_measure_supported": bool(
                        catalog[race_key]["race_type"] == "ballot_measure"
                    ),
                }
            )
        return pl.DataFrame(rows)
