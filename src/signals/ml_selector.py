import os
import datetime
import logging
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Route LightGBM's internal logger through Python logging (respects our level settings)
lgb.register_logger(logging.getLogger("lightgbm"))

from config.settings import MODEL_PATH, ML_MIN_TRAIN_WEEKS, ML_TOP_N, DATA_DIR
from src.signals.feature_engineer import FEATURE_COLS

logger = logging.getLogger(__name__)

_MODEL_MAX_AGE_HOURS = 24


def _make_model(scale_pos_weight: float = 1.0) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )


class MLSelector:
    def __init__(self):
        self.model: LGBMClassifier | None = None
        self._feature_importances: pd.Series | None = None

    def _available_features(self, df: pd.DataFrame) -> list:
        return [c for c in FEATURE_COLS if c in df.columns]

    def train(self, matrix_labeled: pd.DataFrame) -> dict:
        """
        Walk-forward training on the labeled feature matrix.
        Stores the last fold's model to disk. Returns out-of-sample AUC scores.
        matrix_labeled must contain columns: date, symbol, label, + feature cols.
        """
        matrix_labeled = matrix_labeled.dropna(subset=["label"])
        dates = sorted(matrix_labeled["date"].unique())
        weeks = pd.Series(dates).dt.to_period("W").unique()

        min_train = ML_MIN_TRAIN_WEEKS
        if len(weeks) <= min_train + 1:
            logger.warning("Not enough history for walk-forward (%d weeks). Training on full set.", len(weeks))
            return self._train_full(matrix_labeled)

        oos_scores = []
        feat_cols = self._available_features(matrix_labeled)

        for fold_idx in range(min_train, len(weeks) - 1):
            train_cutoff = weeks[fold_idx].end_time.date()
            test_cutoff = weeks[fold_idx + 1].end_time.date()

            train = matrix_labeled[matrix_labeled["date"].dt.date <= train_cutoff]
            test = matrix_labeled[
                (matrix_labeled["date"].dt.date > train_cutoff) &
                (matrix_labeled["date"].dt.date <= test_cutoff)
            ]

            if train.empty or test.empty:
                continue

            X_train = train[feat_cols].fillna(0)
            y_train = train["label"]
            X_test = test[feat_cols].fillna(0)
            y_test = test["label"]

            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            spw = n_neg / n_pos if n_pos > 0 else 1.0

            model = _make_model(scale_pos_weight=spw)
            model.fit(X_train, y_train)

            if y_test.nunique() > 1:
                proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
                oos_scores.append(auc)

            self.model = model

        if self.model:
            os.makedirs(DATA_DIR, exist_ok=True)
            joblib.dump(self.model, MODEL_PATH)
            feat_imp = pd.Series(
                self.model.feature_importances_,
                index=feat_cols,
            ).sort_values(ascending=False)
            self._feature_importances = feat_imp
            logger.info("Model saved. OOS AUC: mean=%.3f over %d folds", np.mean(oos_scores) if oos_scores else 0, len(oos_scores))

        return {"oos_auc_scores": oos_scores, "mean_auc": float(np.mean(oos_scores)) if oos_scores else 0.0}

    def _train_full(self, matrix_labeled: pd.DataFrame) -> dict:
        feat_cols = self._available_features(matrix_labeled)
        X = matrix_labeled[feat_cols].fillna(0)
        y = matrix_labeled["label"]
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        self.model = _make_model(scale_pos_weight=spw)
        self.model.fit(X, y, verbose=False)
        os.makedirs(DATA_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        logger.info("Trained on full dataset (%d rows)", len(X))
        return {"oos_auc_scores": [], "mean_auc": 0.0}

    def load_or_train(self, matrix_labeled: pd.DataFrame) -> dict:
        if os.path.exists(MODEL_PATH):
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
            age_h = (datetime.datetime.now() - mtime).total_seconds() / 3600
            if age_h < _MODEL_MAX_AGE_HOURS:
                self.model = joblib.load(MODEL_PATH)
                logger.info("Loaded model from cache (%.1f hours old)", age_h)
                return {"loaded_from_cache": True}
        return self.train(matrix_labeled)

    def score(self, features_today: pd.DataFrame) -> pd.Series:
        """
        Score today's feature rows. Returns Series indexed by symbol, values = buy probability.
        features_today: DataFrame with columns symbol + feature cols (one row per symbol, latest date).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load_or_train() first.")
        feat_cols = self._available_features(features_today)
        X = features_today.set_index("symbol")[feat_cols].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=X.index, name="ml_score")

    def get_top_n(self, scores: pd.Series, n: int = ML_TOP_N) -> list:
        return scores.nlargest(n).index.tolist()

    @property
    def feature_importances(self) -> pd.Series | None:
        return self._feature_importances
