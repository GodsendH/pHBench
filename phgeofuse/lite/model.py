from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from phgeofuse.cache import atomic_json
from phgeofuse.retrieval import RETRIEVAL_FEATURE_NAMES


DESIGNS = ("P", "P+R", "P+R+M")
ALPHAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
METADATA_NAMES = (
    "plddt_fraction", "log_length", "log_train_organism_count", "non_alphafold_db",
    *(f"ec_{index}" for index in range(1, 8)), "ec_unknown",
)


def feature_names(design: str, embedding_dim: int) -> list[str]:
    if design not in DESIGNS or embedding_dim < 1:
        raise ValueError("invalid Lite design or embedding dimension")
    names = [f"P.{statistic}.{index}" for statistic in ("mean", "std")
             for index in range(embedding_dim)]
    if "R" in design:
        names.extend(f"R.{name}" for name in RETRIEVAL_FEATURE_NAMES)
    if "M" in design:
        names.extend(f"M.{name}" for name in METADATA_NAMES)
    return names


def matrix(value, width: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != width or not len(result):
        raise ValueError(f"expected a nonempty feature matrix with {width} columns")
    if not np.isfinite(result).all():
        raise ValueError("nonfinite features")
    return result


def labels(value, count: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError("labels must be finite and match feature rows")
    return result


def metrics(observed, predicted) -> dict[str, float | int]:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = labels(observed, len(predicted))
    if predicted.shape != observed.shape or not np.isfinite(predicted).all():
        raise ValueError("invalid predictions")
    error = predicted - observed
    return {"count": len(error), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(error.mean())}


@dataclass
class PHGeoFuseLite:
    design: str
    embedding_dim: int
    coefficient: np.ndarray
    intercept: float
    alpha: float
    validation_scores: list[dict[str, float]] = field(default_factory=list)
    metadata_state: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        width = len(self.feature_names)
        self.coefficient = np.asarray(self.coefficient, dtype=np.float64)
        if self.coefficient.shape != (width,) or not np.isfinite(self.coefficient).all():
            raise ValueError("invalid Lite coefficients")
        if not np.isfinite(self.intercept) or not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("invalid Lite intercept or alpha")

    @property
    def feature_names(self) -> list[str]:
        return feature_names(self.design, self.embedding_dim)

    @property
    def parameter_count(self) -> int:
        return len(self.coefficient) + 1

    @classmethod
    def fit(cls, train_x, train_y, validation_x, validation_y, *, design="P+R",
            embedding_dim=1280, alphas=ALPHAS, metadata_state=None, provenance=None):
        width = len(feature_names(design, embedding_dim))
        train_x, validation_x = matrix(train_x, width), matrix(validation_x, width)
        train_y, validation_y = labels(train_y, len(train_x)), labels(validation_y, len(validation_x))
        grid = sorted(set(float(alpha) for alpha in alphas))
        if not grid or any(not np.isfinite(alpha) or alpha <= 0 for alpha in grid):
            raise ValueError("Ridge alphas must be finite and positive")
        best, scores = None, []
        for alpha in grid:
            ridge = Ridge(alpha=alpha, solver="cholesky", fit_intercept=True).fit(train_x, train_y)
            score = metrics(validation_y, ridge.predict(validation_x))["rmse"]
            scores.append({"alpha": alpha, "validation_rmse": score})
            if best is None or score < best[0]:
                best = (score, ridge)
        ridge = best[1]
        return cls(design, embedding_dim, ridge.coef_.copy(), float(ridge.intercept_),
                   float(ridge.alpha), scores, metadata_state or {}, provenance or {})

    def predict(self, features, *, names=None) -> np.ndarray:
        if names is not None and list(names) != self.feature_names:
            raise ValueError("prediction feature schema differs from the fitted model")
        features = matrix(features, len(self.coefficient))
        prediction = features @ self.coefficient + self.intercept
        if not np.isfinite(prediction).all():
            raise ValueError("nonfinite prediction")
        return prediction

    def save(self, destination: str | Path) -> None:
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(f"refusing to replace a Lite model: {destination}")
        atomic_json(destination, {
            "format": "phgeofuse_lite", "schema_version": 1,
            "design": self.design, "embedding_dim": self.embedding_dim,
            "feature_names": self.feature_names, "coefficient": self.coefficient.tolist(),
            "intercept": self.intercept, "alpha": self.alpha,
            "parameter_count": self.parameter_count,
            "validation_scores": self.validation_scores, "metadata_state": self.metadata_state,
            "provenance": self.provenance,
        })

    @classmethod
    def load(cls, source: str | Path) -> "PHGeoFuseLite":
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        if payload.get("format") != "phgeofuse_lite" or payload.get("schema_version") != 1:
            raise ValueError("not a supported PHGeoFuse-Lite model")
        model = cls(**{name: payload[name] for name in (
            "design", "embedding_dim", "coefficient", "intercept", "alpha",
            "validation_scores", "metadata_state", "provenance")})
        if payload["feature_names"] != model.feature_names or payload["parameter_count"] != model.parameter_count:
            raise ValueError("saved Lite feature schema differs")
        return model
