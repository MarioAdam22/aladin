"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — MLflow Experiment Tracking                                        ║
║  mlflow_tracker.py  |  Update #21 — Versionare modele                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Versionează fiecare model antrenat:
  - Parametri (n_estimators, learning_rate etc)
  - Metrici (accuracy, F1, Sharpe pe backtest)
  - Artifacts (model pickle, feature importance plot)

Setup: pip install mlflow
Vizualizare: mlflow ui --port 5000  (http://localhost:5000)
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("aladin-mlflow")

EXPERIMENT_NAME = "Aladin-Quantum-ICT"


def log_training_run(
    model_params:   dict,
    train_metrics:  dict,
    feature_list:   list,
    model_path:     Optional[str] = None,
    feat_imp_path:  Optional[str] = None,
    run_name:       Optional[str] = None,
) -> Optional[str]:
    """
    Update #21: Loghează un run de antrenare în MLflow.

    Args:
        model_params:   Hyperparameters XGBoost/LightGBM
        train_metrics:  accuracy, f1_long, f1_short, sniper_pct etc
        feature_list:   lista de features folosite
        model_path:     path la modelul salvat (.json/.pkl)
        feat_imp_path:  path la CSV feature importance
        run_name:       nume custom pentru run

    Returns:
        run_id: str sau None dacă MLflow nu e disponibil
    """
    try:
        import mlflow
        import mlflow.xgboost

        mlflow.set_experiment(EXPERIMENT_NAME)

        run_name = run_name or f"aladin-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            # Log parametri
            mlflow.log_params(model_params)
            mlflow.log_param("n_features", len(feature_list))
            mlflow.log_param("features", ",".join(feature_list[:20]))  # primele 20

            # Log metrici
            mlflow.log_metrics(train_metrics)
            mlflow.log_metric("timestamp_epoch", datetime.now().timestamp())

            # Log artifacts
            if model_path and os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="models")
            if feat_imp_path and os.path.exists(feat_imp_path):
                mlflow.log_artifact(feat_imp_path, artifact_path="analysis")

            # Tag-uri
            mlflow.set_tag("system",    "Aladin-Quantum-ICT")
            mlflow.set_tag("version",   "6.8")
            mlflow.set_tag("developer", "Adam Mario")

            run_id = run.info.run_id
            logger.info(f"✅ MLflow run loggat: {run_id}")
            print(f"   ✅ MLflow run: {run_id}")
            print(f"   📊 Vizualizare: mlflow ui --port 5000")
            return run_id

    except ImportError:
        logger.warning("MLflow nu e instalat. pip install mlflow")
        print("   ⚠️ MLflow lipsă — run neloggat (pip install mlflow)")
        return None
    except Exception as e:
        logger.error(f"MLflow error: {e}")
        return None


def get_best_run() -> Optional[dict]:
    """
    Returnează cel mai bun run din MLflow bazat pe sharpe_ratio.
    """
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()

        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not exp:
            return None

        runs = client.search_runs(
            experiment_ids = [exp.experiment_id],
            order_by       = ["metrics.sharpe_ratio DESC"],
            max_results    = 1,
        )

        if not runs:
            return None

        best = runs[0]
        return {
            "run_id":       best.info.run_id,
            "run_name":     best.info.run_name,
            "sharpe_ratio": best.data.metrics.get("sharpe_ratio", 0),
            "accuracy":     best.data.metrics.get("accuracy", 0),
            "params":       best.data.params,
        }
    except Exception as e:
        logger.warning(f"MLflow get_best_run error: {e}")
        return None


if __name__ == "__main__":
    # Test rapid
    print("📊 Test MLflow tracking...")
    run_id = log_training_run(
        model_params  = {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.015},
        train_metrics = {"accuracy": 0.78, "sharpe_ratio": 1.85, "f1_long": 0.65, "sniper_pct": 12.5},
        feature_list  = ["close", "high", "low", "slope_h1", "poc_level"],
        run_name      = "test-run-v6.8",
    )
    print(f"   Run ID: {run_id}")
