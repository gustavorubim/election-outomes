from __future__ import annotations

from typing import Any

import polars as pl

from election_outcomes.features import FeatureBundle
from election_outcomes.models.polling_bayes import BayesianPollingModel
from election_outcomes.models.polling_kalman import KalmanPollingModel


def resolve_inference_engine(
    config: dict[str, object] | None = None,
    inference_engine: str | None = None,
) -> str:
    config = config or {}
    if inference_engine:
        selected = inference_engine
    elif config.get("_inference_engine"):
        selected = str(config["_inference_engine"])
    else:
        selected = "bayes" if bool(dict(config.get("bayesian", {})).get("enabled")) else "kalman"
    selected = str(selected).lower().strip()
    if selected not in {"kalman", "bayes"}:
        raise ValueError("inference_engine must be 'kalman' or 'bayes'")
    return selected


class PollingModel:
    """Stable polling component facade.

    The Bayesian engine is selected when `bayesian.enabled` is true; the legacy
    Kalman path remains available through runtime config or
    `--inference-engine kalman`.
    """

    def __init__(
        self,
        config: dict[str, object] | None = None,
        as_of: str | None = None,
        inference_engine: str | None = None,
    ) -> None:
        config = config or {}
        selected = resolve_inference_engine(config, inference_engine)
        if selected == "bayes":
            self._impl: KalmanPollingModel | BayesianPollingModel = BayesianPollingModel(
                config=config, as_of=as_of
            )
        elif selected == "kalman":
            self._impl = KalmanPollingModel(config=config, as_of=as_of)
        self.inference_engine = selected

    @property
    def component(self) -> str:
        return self._impl.component

    @property
    def cached_trajectory(self):
        return self._impl.cached_trajectory

    @property
    def cached_house_effects(self):
        return self._impl.cached_house_effects

    def run(self, bundle: FeatureBundle):
        return self._impl.run(bundle)

    def trajectory(self, bundle: FeatureBundle):
        return self._impl.trajectory(bundle)

    def posterior_draws(self, bundle: FeatureBundle):
        if hasattr(self._impl, "posterior_draws"):
            return self._impl.posterior_draws(bundle)  # type: ignore[attr-defined]
        return pl.DataFrame()

    def diagnostics(self, bundle: FeatureBundle | None = None) -> dict[str, Any]:
        if hasattr(self._impl, "diagnostics"):
            return self._impl.diagnostics(bundle)  # type: ignore[attr-defined]
        return {
            "engine": "kalman",
            "fallback_used": None,
        }

    def _bundle_fingerprint(self, bundle: FeatureBundle) -> str:
        return self._impl._bundle_fingerprint(bundle)
