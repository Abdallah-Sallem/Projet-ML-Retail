from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_raw_data

MODELS_DIR = PROJECT_ROOT / "models"


class RetailChurnPredictor:
    """Load preprocessing + model artifacts and serve churn predictions."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or MODELS_DIR
        self.preprocessing_artifact = joblib.load(self.models_dir / "preprocessing_pipeline.joblib")
        self.model = joblib.load(self.models_dir / "best_model.joblib")

        self.preprocessing_pipeline = self.preprocessing_artifact["pipeline"]
        self.expected_columns: List[str] = self.preprocessing_artifact.get("feature_columns", [])
        self.target_column: str = self.preprocessing_artifact.get("target_column", "Churn")
        self.generated_features: List[str] = self.preprocessing_artifact.get("generated_features", [])

    def _to_dataframe(self, input_data: Any) -> pd.DataFrame:
        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()
        if isinstance(input_data, dict):
            return pd.DataFrame([input_data])
        if isinstance(input_data, list):
            return pd.DataFrame(input_data)

        raise ValueError("Input data must be a dict, list of dicts, or pandas DataFrame.")

    def _align_input_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        aligned = df.copy()

        if self.target_column in aligned.columns:
            aligned.drop(columns=[self.target_column], inplace=True)

        for col in self.expected_columns:
            if col not in aligned.columns:
                aligned[col] = np.nan

        return aligned[self.expected_columns]

    def _get_probabilities(self, x_transformed: Any) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x_transformed)[:, 1]

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(x_transformed)
            return 1.0 / (1.0 + np.exp(-scores))

        return self.model.predict(x_transformed).astype(float)

    def predict(self, input_data: Any) -> Dict[str, Any] | List[Dict[str, Any]]:
        raw_df = self._to_dataframe(input_data)
        cleaned_df = clean_raw_data(raw_df)
        aligned_df = self._align_input_columns(cleaned_df)

        transformed = self.preprocessing_pipeline.transform(aligned_df)
        if isinstance(transformed, np.ndarray) and self.generated_features:
            transformed_input: Any = pd.DataFrame(transformed, columns=self.generated_features)
        else:
            transformed_input = transformed

        probabilities = self._get_probabilities(transformed_input)
        classes = (probabilities >= 0.5).astype(int)

        results = [
            {
                "churn_probability": float(round(prob, 6)),
                "predicted_class": int(cls),
            }
            for prob, cls in zip(probabilities, classes)
        ]

        if isinstance(input_data, dict):
            return results[0]
        return results


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Provide JSON payload as first argument. Example: python src/predict.py '{\"Age\": 30}'")

    payload = json.loads(sys.argv[1])
    predictor = RetailChurnPredictor()
    output = predictor.predict(payload)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
