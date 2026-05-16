from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from civic_signal.features import FeatureBundle

CROSS_OFFICE_SCHEMA: dict[str, pl.DataType] = {
    "draw_id": pl.Int64,
    "cycle": pl.Int64,
    "office_type": pl.String,
    "race_count": pl.Int64,
    "option_count": pl.Int64,
    "office_mean_logit": pl.Float64,
    "national_midterm_environment_logit": pl.Float64,
    "office_offset_logit": pl.Float64,
    "office_share_mean": pl.Float64,
    "shared_draw_stream": pl.Boolean,
    "methodology": pl.String,
    "fallback_used": pl.String,
}


@dataclass(frozen=True)
class CrossOfficeResult:
    posterior: pl.DataFrame
    diagnostics: dict[str, Any]


def summarize_cross_office(
    bundle: FeatureBundle,
    posterior_draws: pl.DataFrame,
    config: dict[str, Any] | None = None,
    posterior_diagnostics: dict[str, Any] | None = None,
) -> CrossOfficeResult:
    """Build the Phase 7 shared midterm environment artifact.

    Under the production NUTS default this summarizes a shared fitted
    state-space draw stream; analytic bridge runs remain explicitly labeled.
    """

    frame = _posterior_with_metadata(bundle, posterior_draws)
    offices = (
        sorted(str(value) for value in frame["office_type"].unique())
        if not frame.is_empty()
        else []
    )
    if frame.is_empty() or len(offices) < 2:
        return CrossOfficeResult(
            posterior=pl.DataFrame(schema=CROSS_OFFICE_SCHEMA),
            diagnostics={"status": "not_applicable", "office_count": len(offices)},
        )
    reference = _reference_party_frame(frame)
    office_draws = reference.group_by(["draw_id", "cycle", "office_type"]).agg(
        pl.col("race_id").n_unique().alias("race_count"),
        pl.col("option_id").n_unique().alias("option_count"),
        pl.col("latent_logit").mean().alias("office_mean_logit"),
        pl.col("latent_share").mean().alias("office_share_mean"),
    )
    national = office_draws.group_by(["draw_id", "cycle"]).agg(
        pl.col("office_mean_logit").mean().alias("national_midterm_environment_logit")
    )
    posterior = (
        office_draws.join(national, on=["draw_id", "cycle"], how="left")
        .with_columns(
            (pl.col("office_mean_logit") - pl.col("national_midterm_environment_logit")).alias(
                "office_offset_logit"
            ),
            pl.lit(True).alias("shared_draw_stream"),
            pl.lit(_methodology_label(posterior_diagnostics)).alias("methodology"),
            pl.lit(_fallback_label(posterior_diagnostics), dtype=pl.String).alias("fallback_used"),
        )
        .sort(["draw_id", "office_type"])
        .select(list(CROSS_OFFICE_SCHEMA))
        .with_columns(_stable_float_expr(column) for column in _FLOAT_COLUMNS)
    )
    diagnostics = _diagnostics(posterior, config or {}, posterior_diagnostics)
    return CrossOfficeResult(posterior=posterior, diagnostics=diagnostics)


def _posterior_with_metadata(bundle: FeatureBundle, posterior_draws: pl.DataFrame) -> pl.DataFrame:
    if posterior_draws.is_empty() or bundle.race_catalog.is_empty():
        return pl.DataFrame()
    offices = ["president", "senate", "house", "governor"]
    catalog = (
        bundle.race_catalog.filter(pl.col("office_type").is_in(offices))
        .select(["race_id", "cycle", "office_type"])
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


def _reference_party_frame(frame: pl.DataFrame) -> pl.DataFrame:
    parties = {str(value) for value in frame["party"].drop_nulls().unique().to_list()}
    if "DEM" in parties:
        return frame.filter(pl.col("party") == "DEM")
    first_party = sorted(parties)[0] if parties else None
    return frame.filter(pl.col("party") == first_party) if first_party else frame


def _diagnostics(
    posterior: pl.DataFrame,
    config: dict[str, Any],
    posterior_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    cross_config = dict(dict(config.get("bayesian", {})).get("cross_office", {}))
    sampling = _sampling_metadata(posterior_diagnostics)
    return {
        "status": "fitted",
        "engine": sampling["office_engine"],
        "sampling_engine": sampling["sampling_engine"],
        "state_space_nuts_fitted": sampling["state_space_nuts_fitted"],
        "office_count": int(posterior["office_type"].n_unique()),
        "offices": sorted(str(value) for value in posterior["office_type"].unique()),
        "draw_count": int(posterior["draw_id"].n_unique()),
        "office_offset_prior_sd": float(cross_config.get("office_offset_prior_sd", 0.02)),
        "shared_draw_stream": True,
        "r_hat_max": sampling["r_hat_max"],
        "ess_min": sampling["ess_min"],
        "divergences": sampling["divergences"],
        "fallback_used": sampling["fallback_used"],
    }


_FLOAT_COLUMNS = [column for column, dtype in CROSS_OFFICE_SCHEMA.items() if dtype == pl.Float64]


def _stable_float_expr(column: str) -> pl.Expr:
    rounded = pl.col(column).round(12)
    return pl.when(rounded.abs() < 5e-13).then(0.0).otherwise(rounded).alias(column)


def _methodology_label(posterior_diagnostics: dict[str, Any] | None) -> str:
    if _sampling_metadata(posterior_diagnostics)["state_space_nuts_fitted"]:
        return "numpyro_nuts_cross_office_shared_midterm_environment"
    return "analytic_cross_office_shared_midterm_environment"


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
            "numpyro-nuts-cross-office-decomposition"
            if state_space_nuts_fitted
            else "analytic_cross_office_bridge"
        ),
        "sampling_engine": engine or "unknown",
        "state_space_nuts_fitted": state_space_nuts_fitted,
        "r_hat_max": diagnostics.get("r_hat_max"),
        "ess_min": diagnostics.get("ess_min"),
        "divergences": diagnostics.get("divergences", 0),
        "fallback_used": fallback_used,
    }
