from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from civic_signal.features import FeatureBundle

HOUSE_HIERARCHICAL_SCHEMA: dict[str, pl.DataType] = {
    "race_id": pl.String,
    "cycle": pl.Int64,
    "state": pl.String,
    "district": pl.String,
    "redistricting_era": pl.String,
    "option_id": pl.String,
    "party": pl.String,
    "draw_count": pl.Int64,
    "posterior_share_mean": pl.Float64,
    "posterior_share_p10": pl.Float64,
    "posterior_share_p90": pl.Float64,
    "posterior_logit_mean": pl.Float64,
    "national_house_environment_logit_mean": pl.Float64,
    "era_effect_logit": pl.Float64,
    "state_effect_logit": pl.Float64,
    "district_idiosyncratic_logit": pl.Float64,
    "district_poll_count": pl.Int64,
    "unpolled_district": pl.Boolean,
    "covariance_method": pl.String,
    "methodology": pl.String,
    "fallback_used": pl.String,
}


@dataclass(frozen=True)
class HouseHierarchicalResult:
    posterior: pl.DataFrame
    diagnostics: dict[str, Any]


def summarize_house_hierarchical(
    bundle: FeatureBundle,
    posterior_draws: pl.DataFrame,
    config: dict[str, Any] | None = None,
    posterior_diagnostics: dict[str, Any] | None = None,
) -> HouseHierarchicalResult:
    """Build the Phase 5 House hierarchy decomposition artifact.

    Under the production NUTS default this is a House-specific decomposition of
    the fitted joint state-space draw stream. Analytic bridge runs remain
    explicitly labeled as bridge artifacts.
    """

    config = config or {}
    frame = _posterior_with_metadata(bundle, posterior_draws)
    if frame.is_empty():
        return HouseHierarchicalResult(
            posterior=pl.DataFrame(schema=HOUSE_HIERARCHICAL_SCHEMA),
            diagnostics={"status": "not_applicable", "district_count": 0},
        )
    option_summary = (
        frame.group_by(
            [
                "race_id",
                "cycle",
                "state",
                "district",
                "redistricting_era",
                "option_id",
                "party",
            ]
        )
        .agg(
            pl.col("draw_id").n_unique().alias("draw_count"),
            pl.col("latent_share").mean().alias("posterior_share_mean"),
            pl.col("latent_share").quantile(0.10).alias("posterior_share_p10"),
            pl.col("latent_share").quantile(0.90).alias("posterior_share_p90"),
            pl.col("latent_logit").mean().alias("posterior_logit_mean"),
        )
        .sort(["cycle", "redistricting_era", "state", "district", "party"])
    )
    environment = _environment_frame(frame)
    poll_counts = _district_poll_counts(bundle)
    covariance_method = str(
        dict(dict(config.get("bayesian", {})).get("house", {})).get(
            "covariance_method", "block_diagonal_state_era_with_district_residuals"
        )
    )
    posterior = (
        option_summary.join(environment, on=["cycle", "redistricting_era", "state"], how="left")
        .join(poll_counts, on="race_id", how="left")
        .with_columns(
            pl.col("district_poll_count").fill_null(0).cast(pl.Int64),
            (pl.col("posterior_logit_mean") - pl.col("state_environment_logit_mean")).alias(
                "district_idiosyncratic_logit"
            ),
            pl.lit(covariance_method).alias("covariance_method"),
            pl.lit(_methodology_label(posterior_diagnostics)).alias("methodology"),
            pl.lit(_fallback_label(posterior_diagnostics), dtype=pl.String).alias("fallback_used"),
        )
        .with_columns((pl.col("district_poll_count") == 0).alias("unpolled_district"))
        .select(list(HOUSE_HIERARCHICAL_SCHEMA))
        .with_columns(_stable_float_expr(column) for column in _FLOAT_COLUMNS)
    )
    diagnostics = _diagnostics(posterior, covariance_method, posterior_diagnostics)
    return HouseHierarchicalResult(posterior=posterior, diagnostics=diagnostics)


def _posterior_with_metadata(bundle: FeatureBundle, posterior_draws: pl.DataFrame) -> pl.DataFrame:
    if posterior_draws.is_empty() or bundle.race_catalog.is_empty():
        return pl.DataFrame()
    era_expr = (
        pl.col("redistricting_era").cast(pl.Utf8)
        if "redistricting_era" in bundle.race_catalog.columns
        else pl.lit("unknown", dtype=pl.Utf8)
    )
    state_expr = _coalesced_text_expr(bundle.race_catalog, ["state"], "unknown")
    district_expr = _coalesced_text_expr(bundle.race_catalog, ["geography", "race_id"], "unknown")
    catalog = (
        bundle.race_catalog.filter(pl.col("office_type") == "house")
        .select(
            "race_id",
            "cycle",
            state_expr.alias("state"),
            district_expr.alias("district"),
            era_expr.fill_null("unknown").alias("redistricting_era"),
        )
        .with_columns(pl.col("cycle").cast(pl.Int64))
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
    national = reference.group_by(["cycle"]).agg(
        pl.col("latent_logit").mean().alias("national_house_environment_logit_mean")
    )
    eras = reference.group_by(["cycle", "redistricting_era"]).agg(
        pl.col("latent_logit").mean().alias("era_environment_logit_mean")
    )
    states = reference.group_by(["cycle", "redistricting_era", "state"]).agg(
        pl.col("latent_logit").mean().alias("state_environment_logit_mean")
    )
    return (
        states.join(eras, on=["cycle", "redistricting_era"], how="left")
        .join(national, on="cycle", how="left")
        .with_columns(
            (
                pl.col("era_environment_logit_mean")
                - pl.col("national_house_environment_logit_mean")
            ).alias("era_effect_logit"),
            (pl.col("state_environment_logit_mean") - pl.col("era_environment_logit_mean")).alias(
                "state_effect_logit"
            ),
        )
        .select(
            [
                "cycle",
                "redistricting_era",
                "state",
                "national_house_environment_logit_mean",
                "era_effect_logit",
                "state_effect_logit",
                "state_environment_logit_mean",
            ]
        )
    )


def _reference_party_frame(frame: pl.DataFrame) -> pl.DataFrame:
    parties = {str(value) for value in frame["party"].drop_nulls().unique().to_list()}
    if "DEM" in parties:
        return frame.filter(pl.col("party") == "DEM")
    first_party = sorted(parties)[0] if parties else None
    return frame.filter(pl.col("party") == first_party) if first_party else frame


def _district_poll_counts(bundle: FeatureBundle) -> pl.DataFrame:
    schema = {"race_id": pl.String, "district_poll_count": pl.Int64}
    if bundle.polls.is_empty():
        return pl.DataFrame(schema=schema)
    house_race_ids = (
        bundle.race_catalog.filter(pl.col("office_type") == "house")["race_id"].to_list()
        if "office_type" in bundle.race_catalog.columns
        else []
    )
    if not house_race_ids:
        return pl.DataFrame(schema=schema)
    return (
        bundle.polls.filter(pl.col("race_id").is_in(house_race_ids))
        .group_by("race_id")
        .agg(pl.col("poll_id").n_unique().alias("district_poll_count"))
    )


def _diagnostics(
    posterior: pl.DataFrame,
    covariance_method: str,
    posterior_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    sampling = _sampling_metadata(posterior_diagnostics)
    return {
        "status": "fitted",
        "engine": sampling["office_engine"],
        "sampling_engine": sampling["sampling_engine"],
        "state_space_nuts_fitted": sampling["state_space_nuts_fitted"],
        "district_count": int(posterior["race_id"].n_unique()),
        "redistricting_eras": sorted(
            str(value) for value in posterior["redistricting_era"].unique()
        ),
        "unpolled_district_count": int(
            posterior.filter(pl.col("unpolled_district")).select("race_id").n_unique()
        ),
        "covariance_method": covariance_method,
        "dense_covariance_used": False,
        "era_partition_enforced": True,
        "r_hat_max": sampling["r_hat_max"],
        "ess_min": sampling["ess_min"],
        "divergences": sampling["divergences"],
        "fallback_used": sampling["fallback_used"],
    }


def _coalesced_text_expr(frame: pl.DataFrame, columns: list[str], fallback: str) -> pl.Expr:
    expressions = [pl.col(column).cast(pl.Utf8) for column in columns if column in frame.columns]
    if not expressions:
        return pl.lit(fallback, dtype=pl.Utf8)
    return pl.coalesce([*expressions, pl.lit(fallback, dtype=pl.Utf8)])


_FLOAT_COLUMNS = [
    column for column, dtype in HOUSE_HIERARCHICAL_SCHEMA.items() if dtype == pl.Float64
]


def _stable_float_expr(column: str) -> pl.Expr:
    rounded = pl.col(column).round(12)
    return pl.when(rounded.abs() < 5e-13).then(0.0).otherwise(rounded).alias(column)


def _methodology_label(posterior_diagnostics: dict[str, Any] | None) -> str:
    if _sampling_metadata(posterior_diagnostics)["state_space_nuts_fitted"]:
        return "numpyro_nuts_house_hierarchical_era_state_district"
    return "analytic_house_hierarchical_bridge_era_state_district"


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
            "numpyro-nuts-house-hierarchical-decomposition"
            if state_space_nuts_fitted
            else "analytic_house_hierarchical_bridge"
        ),
        "sampling_engine": engine or "unknown",
        "state_space_nuts_fitted": state_space_nuts_fitted,
        "r_hat_max": diagnostics.get("r_hat_max"),
        "ess_min": diagnostics.get("ess_min"),
        "divergences": diagnostics.get("divergences", 0),
        "fallback_used": fallback_used,
    }
