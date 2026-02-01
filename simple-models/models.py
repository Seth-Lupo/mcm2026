"""
Model classes for predicting vote proportions.

Each model takes external features and outputs a probability vector
(vote proportions summing to 1).
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Set, List

from data import WeekState, DATA_DIR


@dataclass
class Prediction:
    """Model prediction for a week."""
    votes: np.ndarray  # proportions summing to 1
    confidence: float = 1.0
    meta: Optional[Dict[str, Any]] = None


class BaseModel(ABC):
    """Abstract base class for vote prediction models."""

    name: str = "base"
    description: str = "Base model class"

    def supports_event(self, season: int, week: int, contestants: List[str]) -> bool:
        """
        Check if this model can make predictions for the given event.

        Override in subclasses to restrict to specific seasons/events.
        Default: supports all events.
        """
        return True

    @abstractmethod
    def predict(self, state: WeekState, features: Dict[str, np.ndarray]) -> Prediction:
        """
        Predict vote proportions for contestants.

        Args:
            state: Current week state (contestants, judge scores, etc.)
            features: External features dict, keyed by feature name.
                      Each value is array of shape (n_contestants,).

        Returns:
            Prediction with vote proportions.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class UniformModel(BaseModel):
    """Baseline: equal votes for all contestants."""

    name = "uniform"
    description = "Equal votes for all contestants. votes[i] = 1/n. Random baseline."

    def predict(self, state: WeekState, features: Dict[str, np.ndarray]) -> Prediction:
        n = state.n
        return Prediction(votes=np.ones(n) / n)


class JudgeProportionalModel(BaseModel):
    """Votes proportional to judge scores."""

    name = "judge_proportional"
    description = "Votes proportional to judge scores. votes[i] = judge_score[i] / sum(judge_scores). Assumes fans vote like judges."

    def predict(self, state: WeekState, features: Dict[str, np.ndarray]) -> Prediction:
        return Prediction(votes=state.judge_share())


class FeatureProportionalModel(BaseModel):
    """Votes proportional to a single external feature."""

    name = "feature_proportional"
    description = "Votes proportional to external feature (default: 'popularity'). Falls back to uniform if feature missing."

    def __init__(self, feature_key: str = "popularity"):
        self.feature_key = feature_key

    def predict(self, state: WeekState, features: Dict[str, np.ndarray]) -> Prediction:
        if self.feature_key not in features:
            # Fallback to uniform
            return Prediction(votes=np.ones(state.n) / state.n, confidence=0.0)

        vals = features[self.feature_key]
        vals = np.maximum(vals, 0)  # no negatives
        total = vals.sum()
        if total <= 0:
            return Prediction(votes=np.ones(state.n) / state.n, confidence=0.0)

        return Prediction(votes=vals / total)


class LinearCombinationModel(BaseModel):
    """Weighted combination of judge scores and features."""

    name = "linear_combination"
    description = "Weighted mix of judge scores and features. votes = w*judge_share + (1-w)*feature_share. Default w=0.5."

    def __init__(self, judge_weight: float = 0.5, feature_key: str = "popularity"):
        self.judge_weight = judge_weight
        self.feature_key = feature_key

    def predict(self, state: WeekState, features: Dict[str, np.ndarray]) -> Prediction:
        judge_share = state.judge_share()

        if self.feature_key in features:
            feat = features[self.feature_key]
            feat = np.maximum(feat, 0)
            total = feat.sum()
            feat_share = feat / total if total > 0 else np.ones(state.n) / state.n
        else:
            feat_share = np.ones(state.n) / state.n

        w = self.judge_weight
        combined = w * judge_share + (1 - w) * feat_share
        return Prediction(votes=combined / combined.sum())


class SoftmaxModel(BaseModel):
    """Softmax over weighted features."""

    name = "softmax"
    description = "Softmax over weighted sum of features. votes = softmax(sum(w[k]*feature[k])/temperature). Default: judge scores only."

    def __init__(self, temperature: float = 1.0, feature_weights: Optional[Dict[str, float]] = None):
        self.temperature = temperature
        self.feature_weights = feature_weights or {"judge": 1.0}

    def predict(self, state: WeekState, features: Dict[str, np.ndarray]) -> Prediction:
        scores = np.zeros(state.n)

        for key, weight in self.feature_weights.items():
            if key == "judge":
                scores += weight * state.judge_scores
            elif key in features:
                scores += weight * features[key]

        # Softmax
        scores = scores / self.temperature
        scores = scores - scores.max()  # stability
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()

        return Prediction(votes=probs)


class DirichletModel(BaseModel):
    """Sample from Dirichlet centered on feature-based mean."""

    name = "dirichlet"
    description = "Random sample from Dirichlet(alpha) where alpha = mean * concentration * n. Adds stochastic noise to predictions."

    def __init__(self, concentration: float = 10.0, feature_key: str = "popularity"):
        self.concentration = concentration
        self.feature_key = feature_key
        self.rng = np.random.default_rng()

    def predict(self, state: WeekState, features: Dict[str, np.ndarray]) -> Prediction:
        if self.feature_key in features:
            mean = features[self.feature_key]
            mean = np.maximum(mean, 1e-6)
            mean = mean / mean.sum()
        else:
            mean = np.ones(state.n) / state.n

        alpha = mean * self.concentration * state.n
        votes = self.rng.dirichlet(alpha)

        return Prediction(votes=votes)


class WikiPopularityModel(BaseModel):
    """
    Baseline wiki model: votes proportional to Wikipedia year_total pageviews.

    Only supports seasons 20-34 where wiki data is available.
    """

    name = "wiki_popularity"
    description = "Votes proportional to Wikipedia year_total pageviews. Only S20-34. votes[i] = wiki_views[i] / sum(wiki_views)."

    def __init__(self, column: str = "year_total"):
        self.column = column
        self._wiki_data = None
        self._supported_seasons: Set[int] = set()
        self._load_wiki_data()

    def _load_wiki_data(self) -> None:
        """Load wiki popularity data and build lookup."""
        wiki_path = DATA_DIR / "wiki.csv"
        if not wiki_path.exists():
            return

        df = pd.read_csv(wiki_path)
        # Only keep rows with valid data
        df = df[df[self.column].notna()].copy()

        if df.empty:
            return

        # Build lookup: (season, name) -> popularity value
        self._wiki_data = {}
        for _, row in df.iterrows():
            key = (int(row["season"]), row["celebrity_name"])
            self._wiki_data[key] = float(row[self.column])

        self._supported_seasons = set(df["season"].unique())

    def supports_event(self, season: int, week: int, contestants: List[str]) -> bool:
        """Only support events where we have wiki data for all contestants."""
        if self._wiki_data is None:
            return False
        if season not in self._supported_seasons:
            return False
        # Check all contestants have wiki data
        for name in contestants:
            if (season, name) not in self._wiki_data:
                return False
        return True

    def _get_wiki_share(self, state: WeekState) -> np.ndarray:
        """Get wiki popularity as proportions."""
        popularities = np.array([
            self._wiki_data.get((state.season, name), 0.0)
            for name in state.contestants
        ])
        total = popularities.sum()
        if total <= 0:
            return np.ones(state.n) / state.n
        return popularities / total

    def predict(self, state: WeekState, features: Dict[str, np.ndarray]) -> Prediction:
        if self._wiki_data is None:
            return Prediction(votes=np.ones(state.n) / state.n, confidence=0.0)

        return Prediction(votes=self._get_wiki_share(state))


class WikiJudgeCombinedModel(WikiPopularityModel):
    """
    Combined model: weighted combination of wiki popularity and judge scores.
    """

    name = "wiki_judge_combined"
    description = "50/50 mix of wiki popularity and judge scores. votes = 0.5*wiki_share + 0.5*judge_share. Only S20-34."

    def __init__(self, wiki_weight: float = 0.5, column: str = "year_total"):
        super().__init__(column=column)
        self.wiki_weight = wiki_weight

    def predict(self, state: WeekState, features: Dict[str, np.ndarray]) -> Prediction:
        if self._wiki_data is None:
            return Prediction(votes=state.judge_share(), confidence=0.0)

        wiki_share = self._get_wiki_share(state)
        judge_share = state.judge_share()

        w = self.wiki_weight
        combined = w * wiki_share + (1 - w) * judge_share
        return Prediction(votes=combined / combined.sum())


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class WikiSeasonModel(WikiPopularityModel):
    """Wiki model using season_mean instead of year_total."""

    name = "wiki_season"
    description = "Votes proportional to Wikipedia season_mean (pageviews during show airing). Only S20-34."

    def __init__(self):
        super().__init__(column="season_mean")


MODELS = {
    "uniform": UniformModel,
    "judge_proportional": JudgeProportionalModel,
    "feature_proportional": FeatureProportionalModel,
    "linear_combination": LinearCombinationModel,
    "softmax": SoftmaxModel,
    "dirichlet": DirichletModel,
    "wiki_popularity": WikiPopularityModel,
    "wiki_season": WikiSeasonModel,
    "wiki_judge_combined": WikiJudgeCombinedModel,
}


def get_model(name: str, **kwargs) -> BaseModel:
    """Factory function to get a model by name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name](**kwargs)
