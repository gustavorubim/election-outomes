from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from election_outcomes.features import FeatureBundle

SENATE_JOINT_SCHEMA: dict[str, pl.DataType] = {
    "race_id": pl.String,
    "cycle": pl.Int64,
    "state": pl.String,
    "senate_class": pl.String,
    "option_id": pl.String,
    "party": pl.String,
    "draw_count": pl.Int64,
    "posterior_share_mean": pl.Float64,
    "posterior_share_p10": pl.Float64,
    "posterior_share_p90": pl.Float64,
    "posterior_logit_mean": pl.Float64,
    "national_environment_logit_mean": pl.Float64,
    "class_effect_logit": pl.Float64,
    "state_deviation_logit": pl.Float64,
    "modeled_seat_count_mean": pl.Float64,
    "total_seat_count_mean": pl.Float64,
    "majority_probability": pl.Float64,
    "majority_threshold": pl.Int64,
    "methodology": pl.String,
    "fallback_used": pl.String,
}


@dataclass(frozen=True)
class SenateJointResult:
    posterior: pl.DataFrame
    diagnostics: dict[str, Any]


def summarize_senate_joint(
    bundle: FeatureBundle,
    posterior_draws: pl.DataFrame,
    seat_posterior: pl.DataFrame | None = None,
    posterior_diagnostics: dict[str, Any] | None = None,
) -> SenateJointResult:
    """Build the Phase 4 Senate joint-state decomposition artifact.

    The summary is backed by whichever Bayesian posterior produced
    ``posterior_draws``. Under the production NUTS default, it is a deterministic
    decomposition of the fitted joint state-space draw stream; under the analytic
    bridge it remains labeled as an analytic bridge artifact.
    """

    frame = _posterior_with_metadata(bundle, posterior_draws)
    if frame.is_empty():
        return SenateJointResult(
            posterior=pl.DataFrame(schema=SENATE_JOINT_SCHEMA),
            diagnostics={"status": "not_applicable", "race_count": 0},
        )
    option_summary = (
        frame.group_by(["race_id", "cycle", "state", "senate_class", "option_id", "party"])
        .agg(
            pl.col("draw_id").n_unique().alias("draw_count"),
            pl.col("latent_share").mean().alias("posterior_share_mean"),
            pl.col("latent_share").quantile(0.10).alias("posterior_share_p10"),
            pl.col("latent_share").quantile(0.90).alias("posterior_share_p90"),
            pl.col("latent_logit").mean().alias("posterior_logit_mean"),
        )
        .sort(["cycle", "state", "race_id", "party"])
    )
    environment = _environment_frame(frame)
    seats = _seat_summary(seat_posterior, control_body="senate")
    posterior = (
        option_summary.join(environment, on=["cycle", "senate_class", "state"], how="left")
        .join(seats, on="party", how="left")
        .with_columns(
            (pl.col("posterior_logit_mean") - pl.col("national_environment_logit_mean")).alias(
                "state_deviation_logit"
            ),
            pl.lit(_methodology_label(posterior_diagnostics)).alias("methodology"),
            pl.lit(_fallback_label(posterior_diagnostics), dtype=pl.String).alias("fallback_used"),
        )
        .select(list(SENATE_JOINT_SCHEMA))
        .with_columns(_stable_float_expr(column) for column in _FLOAT_COLUMNS)
    )
    diagnostics = _diagnostics(posterior, posterior_diagnostics)
    return SenateJointResult(posterior=posterior, diagnostics=diagnostics)


def _posterior_with_metadata(bundle: FeatureBundle, posterior_draws: pl.DataFrame) -> pl.DataFrame:
    if posterior_draws.is_empty() or bundle.race_catalog.is_empty():
        return pl.DataFrame()
    state_expr = _coalesced_text_expr(bundle.race_catalog, ["state", "geography"], "unknown")
    catalog = (
        bundle.race_catalog.filter(pl.col("office_type") == "senate")
        .select("race_id", "cycle", state_expr.alias("state"))
        .with_columns(
            pl.col("state").cast(pl.Utf8),
            pl.col("cycle").cast(pl.Int64),
            pl.col("cycle")
            .map_elements(_senate_class, return_dtype=pl.String)
            .alias("senate_class"),
        )
    )
    if catalog.is_empty():
        return pl.DataFrame()
    options = bundle.options.select(["race_id", "option_id", "party"]).with_columns(
        pl.col("party").cast(pl.Utf8).str.to_uppercase()
    )
    return posterior_draws.join(catalog, on="race_id", how="inner").join(
        options, on=["race_id", "option_id"], how="left"
    )


def _environment_frame(frame: pl.DataFrame) -> pl.DataFrame:
    reference = _reference_party_frame(frame)
    national = reference.group_by("cycle").agg(
        pl.col("latent_logit").mean().alias("national_environment_logit_mean")
    )
    class_effect = reference.group_by(["cycle", "senate_class"]).agg(
        pl.col("latent_logit").mean().alias("class_environment_logit_mean")
    )
    states = reference.group_by(["cycle", "senate_class", "state"]).agg(
        pl.col("latent_logit").mean().alias("state_environment_logit_mean")
    )
    return (
        states.join(class_effect, on=["cycle", "senate_class"], how="left")
        .join(national, on="cycle", how="left")
        .with_columns(
            (
                pl.col("class_environment_logit_mean") - pl.col("national_environment_logit_mean")
            ).alias("class_effect_logit")
        )
        .select(
            [
                "cycle",
                "senate_class",
                "state",
                "national_environment_logit_mean",
                "class_effect_logit",
            ]
        )
    )


def _reference_party_frame(frame: pl.DataFrame) -> pl.DataFrame:
    parties = {str(value) for value in frame["party"].drop_nulls().unique().to_list()}
    if "DEM" in parties:
        return frame.filter(pl.col("party") == "DEM")
    first_party = sorted(parties)[0] if parties else None
    return frame.filter(pl.col("party") == first_party) if first_party else frame


def _seat_summary(seat_posterior: pl.DataFrame | None, control_body: str) -> pl.DataFrame:
    schema = {
        "party": pl.String,
        "modeled_seat_count_mean": pl.Float64,
        "total_seat_count_mean": pl.Float64,
        "majority_probability": pl.Float64,
        "majority_threshold": pl.Int64,
    }
    if seat_posterior is None or seat_posterior.is_empty():
        return pl.DataFrame(schema=schema)
    frame = seat_posterior.filter(pl.col("control_body") == control_body)
    if frame.is_empty():
        return pl.DataFrame(schema=schema)
    return frame.group_by("party").agg(
        pl.col("seat_count_modeled").mean().alias("modeled_seat_count_mean"),
        pl.col("seat_count_total").mean().alias("total_seat_count_mean"),
        pl.col("majority").mean().alias("majority_probability"),
        pl.col("majority_threshold").max().alias("majority_threshold"),
    )


def _diagnostics(
    posterior: pl.DataFrame, posterior_diagnostics: dict[str, Any] | None
) -> dict[str, Any]:
    sampling = _sampling_metadata(posterior_diagnostics)
    return {
        "status": "fitted",
        "engine": sampling["office_engine"],
        "sampling_engine": sampling["sampling_engine"],
        "state_space_nuts_fitted": sampling["state_space_nuts_fitted"],
        "race_count": int(posterior["race_id"].n_unique()),
        "senate_classes": sorted(str(value) for value in posterior["senate_class"].unique()),
        "r_hat_max": sampling["r_hat_max"],
        "ess_min": sampling["ess_min"],
        "divergences": sampling["divergences"],
        "zero_divergences": int(sampling["divergences"] or 0) == 0,
        "shared_environment": "cycle_mean_reference_party_logit",
        "class_effect_prior": "tight_zero_centered_residual",
        "fallback_used": sampling["fallback_used"],
    }


def _senate_class(cycle: int | None) -> str:
    if cycle is None:
        return "unknown"
    return {2: "I", 4: "II", 0: "III"}.get(int(cycle) % 6, "unknown")


def _coalesced_text_expr(frame: pl.DataFrame, columns: list[str], fallback: str) -> pl.Expr:
    expressions = [pl.col(column).cast(pl.Utf8) for column in columns if column in frame.columns]
    if not expressions:
        return pl.lit(fallback, dtype=pl.Utf8)
    return pl.coalesce([*expressions, pl.lit(fallback, dtype=pl.Utf8)])


_FLOAT_COLUMNS = [column for column, dtype in SENATE_JOINT_SCHEMA.items() if dtype == pl.Float64]


def _stable_float_expr(column: str) -> pl.Expr:
    rounded = pl.col(column).round(12)
    return pl.when(rounded.abs() < 5e-13).then(0.0).otherwise(rounded).alias(column)


def _methodology_label(posterior_diagnostics: dict[str, Any] | None) -> str:
    if _sampling_metadata(posterior_diagnostics)["state_space_nuts_fitted"]:
        return "numpyro_nuts_senate_joint_shared_environment_class_effect"
    return "analytic_senate_joint_bridge_shared_environment_class_effect"


def _fallback_label(posterior_diagnostics: dict[str, Any] | None) -> str | None:
    fallback = _sampling_metadata(posterior_diagnostics)["fallback_used"]
    return str(fallback) if fallback else None


def _sampling_metadata(posterior_diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    diagnostics = posterior_diagnostics or {}
    engine = str(diagnostics.get("engine") or "")
    fallback_used = diagnostics.get("fallback_used")
    state_space_nuts_fitted = engine == "numpyro-nuts" and not fallback_used
    return {
        "office_engine": (
            "numpyro-nuts-senate-joint-decomposition"
            if state_space_nuts_fitted
            else "analytic_senate_joint_bridge"
        ),
        "sampling_engine": engine or "unknown",
        "state_space_nuts_fitted": state_space_nuts_fitted,
        "r_hat_max": diagnostics.get("r_hat_max"),
        "ess_min": diagnostics.get("ess_min"),
        "divergences": diagnostics.get("divergences", 0),
        "fallback_used": fallback_used,
    }
