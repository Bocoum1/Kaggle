"""Selection de modeles House Prices par validation croisee, puis test interne separe."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


#============= 1) Donnees et pretraitements ====================

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_data(path):
    data = pd.read_csv(path)
    if not {"Id", "SalePrice"}.issubset(data.columns) or not data["Id"].is_unique:
        raise ValueError("Il faut un Id unique par maison et la cible SalePrice.")
    if not np.isfinite(data["SalePrice"]).all() or (data["SalePrice"] <= 0).any():
        raise ValueError("SalePrice doit etre strictement positif et sans valeur manquante.")
    # Id est un identifiant, pas une mesure du logement. Aucun retrait d'outlier selon son prix.
    features = data.drop(columns=["Id", "SalePrice"])
    if "MSSubClass" in features:
        # Le code numerique designe une categorie de logement, pas une quantite continue.
        features["MSSubClass"] = features["MSSubClass"].astype(str)
    return data, features


def make_pipeline(features, estimator):
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [col for col in features.columns if col not in numeric]
    # La mediane et la standardisation sont apprises dans fit(), donc dans chaque pli.
    numeric_steps = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scale", StandardScaler())])
    # Une modalite inconnue au moment de predire ne fait pas echouer l'encodage.
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    processor = ColumnTransformer([("numeric", numeric_steps, numeric),
                                    ("categorical", categorical_steps, categorical)])
    return Pipeline([("preprocess", processor), ("model", estimator)])


def candidates(seed):
    # Petite grille declaree a l'avance : pas de recherche supplementaire apres lecture du test.
    return {
        "baseline": (DummyRegressor(strategy="mean"), {}),
        "ridge": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}),
        "random_forest": (RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1),
                          {"model__min_samples_leaf": [1, 3], "model__max_features": [0.7, 1.0]}),
        "gradient_boosting": (GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                        min_samples_leaf=3, random_state=seed),
                              {"model__max_depth": [2, 3]}),
    }


#============= 2) Comparaison par validation croisee ====================

def train_experiment(args):
    if args.output.exists():
        raise FileExistsError("Choisir un nouveau dossier pour ne pas ecraser une experience.")
    data, features = load_data(args.train_csv)
    development, test = train_test_split(np.arange(len(data)), test_size=0.2, random_state=args.seed)
    # Le test interne reste a l'ecart de GridSearchCV et du choix de modele.
    x_dev = features.iloc[development]
    log_y = np.log(data["SalePrice"].iloc[development])
    folds = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(x_dev))
    args.output.mkdir(parents=True)
    np.savez(args.output / "splits.npz", development=development, test=test)
    # Enregistrer les affectations aux plis rend les comparaisons verifiables ligne par ligne.
    fold_ids = np.empty(len(development), dtype=int)
    for fold, (_, validation) in enumerate(folds):
        fold_ids[validation] = fold
    pd.DataFrame({"row_index": development, "fold": fold_ids}).to_csv(args.output / "folds.csv", index=False)
    manifest = {
        "dataset_sha256": digest(args.train_csv), "code_sha256": digest(__file__), "seed": args.seed,
        "development_size": len(development), "test_size": len(test), "folds": 5,
        "target": "natural log of SalePrice", "selection": "minimum mean CV RMSE on log prices",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    save_json(args.output / "protocol.json", manifest)
    summaries, fitted, grids = [], {}, {}
    for name, (estimator, grid) in candidates(args.seed).items():
        search = GridSearchCV(make_pipeline(x_dev, estimator), grid, cv=folds,
                              scoring="neg_root_mean_squared_error", return_train_score=True,
                              refit=True, error_score="raise", n_jobs=1)
        start = time.perf_counter()
        search.fit(x_dev, log_y)
        best = search.best_index_
        summaries.append({"model": name, "cv_rmse_log": -search.best_score_,
                          "cv_std": search.cv_results_["std_test_score"][best],
                          "train_rmse_log": -search.cv_results_["mean_train_score"][best],
                          "search_seconds": time.perf_counter() - start})
        fitted[name], grids[name] = search.best_estimator_, search.best_params_
        pd.DataFrame(search.cv_results_).to_csv(args.output / f"search-{name}.csv", index=False)
        print(f"{name}: CV RMSE(log)={-search.best_score_:.4f}; {search.best_params_}", flush=True)
    comparison = pd.DataFrame(summaries).sort_values("cv_rmse_log")
    comparison.to_csv(args.output / "validation.csv", index=False)
    selected = comparison.iloc[0]["model"]
    manifest.update(selected_model=selected, best_parameters=grids)
    save_json(args.output / "protocol.json", manifest)
    # Seuls le modele retenu et la reference seront evalues sur le test final.
    for name in dict.fromkeys(["baseline", selected]):
        joblib.dump(fitted[name], args.output / f"{name}.joblib")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(comparison.model, comparison.cv_rmse_log, xerr=comparison.cv_std, color="#176076")
    ax.set(xlabel="RMSE sur log(prix), plus bas = meilleur", title="Validation croisee : moyenne et ecart-type des 5 plis")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=150)
    plt.close(fig)
    print(f"Modele retenu : {selected}. Test non evalue.")


#============= 3) Test final et diagnostic des erreurs ====================

def regression_metrics(observed, predicted_log):
    predicted = np.exp(predicted_log)
    return {"rmse_log": root_mean_squared_error(np.log(observed), predicted_log),
            "mae_dollars": mean_absolute_error(observed, predicted),
            "rmse_dollars": root_mean_squared_error(observed, predicted),
            "r2": r2_score(observed, predicted)}


def evaluate_experiment(args):
    manifest = json.loads((args.output / "protocol.json").read_text())
    if (args.output / "test-metrics.csv").exists():
        raise FileExistsError("Le test a deja ete evalue pour cette experience.")
    if manifest["dataset_sha256"] != digest(args.train_csv) or manifest["code_sha256"] != digest(__file__):
        raise ValueError("Donnees ou code differents de ceux utilises pour la selection.")
    data, features = load_data(args.train_csv)
    test = np.load(args.output / "splits.npz")["test"]
    observed = data["SalePrice"].iloc[test].to_numpy()
    selected, metrics = manifest["selected_model"], []
    for name in dict.fromkeys(["baseline", selected]):
        # Ne charger que les fichiers produits localement : joblib n'est pas un format sur pour des tiers.
        model = joblib.load(args.output / f"{name}.joblib")
        predicted_log = model.predict(features.iloc[test])
        metrics.append({"model": name, **regression_metrics(observed, predicted_log)})
        if name == selected:
            diagnostics = pd.DataFrame({"Id": data["Id"].iloc[test].to_numpy(),
                                        "observed": observed, "predicted": np.exp(predicted_log)})
            diagnostics["residual"] = diagnostics.observed - diagnostics.predicted
            diagnostics["absolute_error"] = diagnostics.residual.abs()
            diagnostics.to_csv(args.output / "test-predictions.csv", index=False)
            plot_errors(diagnostics, args.output)
    results = pd.DataFrame(metrics)
    results.to_csv(args.output / "test-metrics.csv", index=False)
    print(results.to_string(index=False))


def plot_errors(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(data.observed, data.predicted, alpha=0.6, s=15)
    limits = [min(data.observed.min(), data.predicted.min()), max(data.observed.max(), data.predicted.max())]
    axes[0].plot(limits, limits, linestyle="--", color="black")
    axes[0].set(xlabel="Prix observe ($)", ylabel="Prix predit ($)", title="Modele retenu : test interne")
    axes[1].scatter(data.predicted, data.residual, alpha=0.6, s=15)
    axes[1].axhline(0, linestyle="--", color="black")
    axes[1].set(xlabel="Prix predit ($)", ylabel="Observe - predit ($)", title="Residus : dispersion et erreurs extremes")
    fig.tight_layout()
    fig.savefig(output / "residuals.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "evaluate"])
    parser.add_argument("--train-csv", type=Path, default=Path("data/house-prices/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/house-prices"))
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    (train_experiment if args.stage == "train" else evaluate_experiment)(args)


if __name__ == "__main__":
    main()
