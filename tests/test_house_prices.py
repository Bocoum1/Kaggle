import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from house_prices.experiment import evaluate_experiment, load_data, make_pipeline, regression_metrics, train_experiment


class HousePricesTest(unittest.TestCase):
    def test_existing_experiment_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                train_experiment(SimpleNamespace(output=Path(directory)))

    def test_test_evaluation_is_guarded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "protocol.json").write_text("{}")
            (output / "test-metrics.csv").touch()
            with self.assertRaises(FileExistsError):
                evaluate_experiment(SimpleNamespace(output=output))

    # Le pretraitement doit garder la mediane du train, meme si le test contient des valeurs extremes.
    def test_preprocessing_is_fitted_only_on_train(self):
        train = pd.DataFrame({"area": [1.0, 3.0, np.nan, 5.0], "district": ["a", "b", "a", "b"]})
        model = make_pipeline(train, Ridge()).fit(train, [1, 2, 1, 3])
        imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        self.assertEqual(imputer.statistics_[0], 3.0)
        prediction = model.predict(pd.DataFrame({"area": [10000, np.nan], "district": ["unknown", "a"]}))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(imputer.statistics_[0], 3.0)

    def test_metrics_return_to_dollars(self):
        metrics = regression_metrics(np.array([100.0, 200.0]), np.log([100.0, 200.0]))
        self.assertAlmostEqual(metrics["mae_dollars"], 0.0)
        self.assertAlmostEqual(metrics["rmse_log"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)

    def test_identifiers_and_target_are_not_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            pd.DataFrame({"Id": [1, 2], "SalePrice": [100, 200], "MSSubClass": [20, 30]}).to_csv(path, index=False)
            _, features = load_data(path)
            self.assertEqual(list(features.columns), ["MSSubClass"])
            self.assertEqual(features["MSSubClass"].tolist(), ["20", "30"])


if __name__ == "__main__":
    unittest.main()
