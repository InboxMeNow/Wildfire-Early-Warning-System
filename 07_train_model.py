#!/usr/bin/env python3
"""
Train advanced Spark MLlib wildfire occurrence models.

Input:
    s3a://wildfire-data/features/

Outputs:
    s3a://wildfire-data/models/rf_baseline/
    s3a://wildfire-data/models/rf_tuned/
    s3a://wildfire-data/models/gbt/
    reports/model_metrics_week1.json
    reports/model_comparison_week1.csv
    reports/feature_importance_week1.csv
    reports/feature_importance_week1.png
    reports/calibration_curves_week1.csv
    reports/calibration_curves_week1.png
    reports/threshold_optimization_week1.csv
    reports/threshold_optimization_week1.png
    reports/mlruns/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassificationModel,
    GBTClassifier,
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
DEFAULT_MINIO_ACCESS_KEY = "minioadmin"
DEFAULT_MINIO_SECRET_KEY = "minioadmin"
DEFAULT_MINIO_BUCKET = "wildfire-data"
DEFAULT_HADOOP_AWS_PACKAGE = "org.apache.hadoop:hadoop-aws:3.3.4"
REGISTERED_MODEL_NAMES = {
    "rf_tuned": "wildfire-rf-tuned",
    "gbt": "wildfire-gbt",
}

BASE_FEATURE_COLUMNS = [
    "grid_lat_index",
    "grid_lon_index",
    "grid_lat",
    "grid_lon",
    "temperature_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "precipitation_sum",
    "precipitation_sum_7days",
    "dry_days_count",
    "mean_ndvi_30days",
    "station_distance_km",
    "weather_points_count",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]

ADVANCED_FEATURE_COLUMNS = [
    "precip_lag_1",
    "precip_lag_3",
    "precip_lag_7",
    "temp_lag_1",
    "temp_lag_3",
    "temp_lag_7",
    "humidity_lag_1",
    "wind_lag_1",
    "ndvi_lag_1",
    "temp_7d_avg",
    "temp_7d_std",
    "temp_30d_avg",
    "temp_30d_std",
    "humidity_7d_avg",
    "humidity_30d_avg",
    "wind_7d_avg",
    "precip_14d_sum",
    "precip_30d_sum",
    "dry_days_14d",
    "dry_days_30d",
    "ndvi_30d_avg",
    "ndvi_30d_min",
    "heat_dryness_index",
    "wind_dryness_index",
    "vegetation_dryness_index",
    "lat_band",
    "lon_band",
]

DEFAULT_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ADVANCED_FEATURE_COLUMNS


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def s3a_path(bucket: str, prefix: str) -> str:
    return f"s3a://{bucket}/{prefix.strip('/')}/"


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_spark(args: argparse.Namespace) -> SparkSession:
    builder = SparkSession.builder.appName(args.spark_app_name)

    if args.spark_master:
        builder = builder.master(args.spark_master)

    if args.hadoop_aws_package:
        builder = builder.config("spark.jars.packages", args.hadoop_aws_package)

    builder = (
        builder.config("spark.hadoop.fs.s3a.endpoint", args.minio_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", args.minio_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", args.minio_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", str(args.minio_endpoint.startswith("https://")).lower())
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.sql.session.timeZone", args.timezone)
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.default.parallelism", str(args.shuffle_partitions))
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.executor.memory", args.executor_memory)
    )

    return builder.getOrCreate()


def add_time_features(frame: DataFrame) -> DataFrame:
    month = F.month("date")
    dayofyear = F.dayofyear("date")
    return (
        frame.withColumn("month", month.cast("double"))
        .withColumn("dayofyear", dayofyear.cast("double"))
        .withColumn("month_sin", F.sin(2 * F.pi() * month / F.lit(12.0)))
        .withColumn("month_cos", F.cos(2 * F.pi() * month / F.lit(12.0)))
        .withColumn("dayofyear_sin", F.sin(2 * F.pi() * dayofyear / F.lit(366.0)))
        .withColumn("dayofyear_cos", F.cos(2 * F.pi() * dayofyear / F.lit(366.0)))
    )


def add_advanced_features(frame: DataFrame) -> DataFrame:
    frame = add_time_features(frame)
    grid_by_date = Window.partitionBy("grid_id").orderBy("date")
    prev_7 = grid_by_date.rowsBetween(-7, -1)
    prev_14 = grid_by_date.rowsBetween(-14, -1)
    prev_30 = grid_by_date.rowsBetween(-30, -1)

    is_dry_day = F.when(F.col("precipitation_sum") < F.lit(1.0), F.lit(1.0)).otherwise(F.lit(0.0))
    return (
        frame.withColumn("precip_lag_1", F.lag("precipitation_sum", 1).over(grid_by_date))
        .withColumn("precip_lag_3", F.lag("precipitation_sum", 3).over(grid_by_date))
        .withColumn("precip_lag_7", F.lag("precipitation_sum", 7).over(grid_by_date))
        .withColumn("temp_lag_1", F.lag("temperature_2m_max", 1).over(grid_by_date))
        .withColumn("temp_lag_3", F.lag("temperature_2m_max", 3).over(grid_by_date))
        .withColumn("temp_lag_7", F.lag("temperature_2m_max", 7).over(grid_by_date))
        .withColumn("humidity_lag_1", F.lag("relative_humidity_2m_min", 1).over(grid_by_date))
        .withColumn("wind_lag_1", F.lag("wind_speed_10m_max", 1).over(grid_by_date))
        .withColumn("ndvi_lag_1", F.lag("mean_ndvi_30days", 1).over(grid_by_date))
        .withColumn("temp_7d_avg", F.avg("temperature_2m_max").over(prev_7))
        .withColumn("temp_7d_std", F.stddev("temperature_2m_max").over(prev_7))
        .withColumn("temp_30d_avg", F.avg("temperature_2m_max").over(prev_30))
        .withColumn("temp_30d_std", F.stddev("temperature_2m_max").over(prev_30))
        .withColumn("humidity_7d_avg", F.avg("relative_humidity_2m_min").over(prev_7))
        .withColumn("humidity_30d_avg", F.avg("relative_humidity_2m_min").over(prev_30))
        .withColumn("wind_7d_avg", F.avg("wind_speed_10m_max").over(prev_7))
        .withColumn("precip_14d_sum", F.sum("precipitation_sum").over(prev_14))
        .withColumn("precip_30d_sum", F.sum("precipitation_sum").over(prev_30))
        .withColumn("dry_days_14d", F.sum(is_dry_day).over(prev_14))
        .withColumn("dry_days_30d", F.sum(is_dry_day).over(prev_30))
        .withColumn("ndvi_30d_avg", F.avg("mean_ndvi_30days").over(prev_30))
        .withColumn("ndvi_30d_min", F.min("mean_ndvi_30days").over(prev_30))
        .withColumn(
            "heat_dryness_index",
            F.col("temperature_2m_max") * (F.lit(100.0) - F.col("relative_humidity_2m_min")),
        )
        .withColumn(
            "wind_dryness_index",
            F.col("wind_speed_10m_max") * (F.lit(100.0) - F.col("relative_humidity_2m_min")),
        )
        .withColumn(
            "vegetation_dryness_index",
            F.coalesce(F.col("mean_ndvi_30days"), F.lit(0.0)) * F.coalesce(F.col("dry_days_30d"), F.lit(0.0)),
        )
        .withColumn("lat_band", F.floor(F.col("grid_lat") / F.lit(5.0)).cast("double"))
        .withColumn("lon_band", F.floor(F.col("grid_lon") / F.lit(5.0)).cast("double"))
    )


def ensure_feature_columns(frame: DataFrame, feature_columns: list[str]) -> DataFrame:
    result = frame
    for column in feature_columns:
        if column not in result.columns:
            result = result.withColumn(column, F.lit(0.0).cast("double"))
    return result


def prepare_data(frame: DataFrame, feature_columns: list[str]) -> DataFrame:
    frame = add_advanced_features(frame)
    frame = ensure_feature_columns(frame, feature_columns)
    return frame.fillna(0.0, subset=feature_columns)


def add_class_weights(train_df: DataFrame, label_col: str = "fire_occurred") -> tuple[DataFrame, dict[str, float]]:
    counts = {
        int(row[label_col]): int(row["count"])
        for row in train_df.groupBy(label_col).count().collect()
    }
    negative_count = counts.get(0, 0)
    positive_count = counts.get(1, 0)
    total = negative_count + positive_count

    if not negative_count or not positive_count:
        raise ValueError(f"Both classes are required in training data, got counts={counts}")

    negative_weight = total / (2.0 * negative_count)
    positive_weight = total / (2.0 * positive_count)

    weighted = train_df.withColumn(
        "class_weight",
        F.when(F.col(label_col) == F.lit(1), F.lit(positive_weight)).otherwise(F.lit(negative_weight)),
    )
    return weighted, {
        "negative_count": negative_count,
        "positive_count": positive_count,
        "negative_weight": negative_weight,
        "positive_weight": positive_weight,
    }


def add_positive_probability(predictions: DataFrame) -> DataFrame:
    return predictions.withColumn("positive_probability", vector_to_array("probability")[1].cast("double"))


def collect_scores(predictions: DataFrame) -> list[tuple[int, float]]:
    scored = add_positive_probability(predictions).select(
        F.col("fire_occurred").cast("int").alias("label"),
        F.col("positive_probability").cast("double").alias("probability"),
    )
    return [(int(row["label"]), float(row["probability"])) for row in scored.collect()]


def binary_metrics_from_scores(scores: list[tuple[int, float]], threshold: float) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for label, probability in scores:
        predicted = 1 if probability >= threshold else 0
        if label == 1 and predicted == 1:
            tp += 1
        elif label == 0 and predicted == 1:
            fp += 1
        elif label == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
    }


def precision_recall_at_k(scores: list[tuple[int, float]], fraction: float) -> dict[str, float | int]:
    if not scores:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "k": 0}

    k = max(1, int(round(len(scores) * fraction)))
    ordered = sorted(scores, key=lambda item: item[1], reverse=True)
    top_k = ordered[:k]
    positive_count = sum(1 for label, _ in scores if label == 1)
    true_positive_at_k = sum(1 for label, _ in top_k if label == 1)

    return {
        "precision_at_k": true_positive_at_k / k if k else 0.0,
        "recall_at_k": true_positive_at_k / positive_count if positive_count else 0.0,
        "k": k,
    }


def threshold_curve(scores: list[tuple[int, float]], thresholds: list[float]) -> list[dict[str, float | int]]:
    rows = []
    for threshold in thresholds:
        metrics = binary_metrics_from_scores(scores, threshold)
        rows.append({"threshold": threshold, **metrics})
    return rows


def choose_best_threshold(rows: list[dict[str, float | int]]) -> tuple[float, dict[str, float | int]]:
    if not rows:
        return 0.5, {}

    best = max(rows, key=lambda row: (float(row["f1"]), float(row["recall"]), -float(row["false_positive"])))
    return float(best["threshold"]), best


def calibration_bins(scores: list[tuple[int, float]], bins: int = 10) -> list[dict[str, float | int]]:
    buckets: list[list[tuple[int, float]]] = [[] for _ in range(bins)]
    for label, probability in scores:
        index = min(bins - 1, max(0, int(probability * bins)))
        buckets[index].append((label, probability))

    rows = []
    for index, bucket in enumerate(buckets):
        lower = index / bins
        upper = (index + 1) / bins
        count = len(bucket)
        if count:
            mean_predicted = sum(probability for _, probability in bucket) / count
            actual_rate = sum(label for label, _ in bucket) / count
        else:
            continue
        rows.append(
            {
                "bin": index,
                "probability_min": lower,
                "probability_max": upper,
                "count": count,
                "mean_predicted_probability": mean_predicted,
                "actual_fire_rate": actual_rate,
            }
        )
    return rows


def extract_stage_model(model, model_type):
    for stage in model.stages:
        if isinstance(stage, model_type):
            return stage
    raise RuntimeError(f"PipelineModel does not contain {model_type.__name__}")


def make_assembler(feature_columns: list[str]) -> VectorAssembler:
    return VectorAssembler(
        inputCols=feature_columns,
        outputCol="features",
        handleInvalid="keep",
    )


def fit_rf_baseline(train_df: DataFrame, feature_columns: list[str], args: argparse.Namespace):
    rf = RandomForestClassifier(
        labelCol="fire_occurred",
        featuresCol="features",
        weightCol="class_weight",
        numTrees=args.rf_baseline_num_trees,
        maxDepth=args.rf_baseline_max_depth,
        minInstancesPerNode=args.rf_baseline_min_instances_per_node,
        featureSubsetStrategy=args.rf_baseline_feature_subset_strategy,
        seed=args.seed,
    )
    return Pipeline(stages=[make_assembler(feature_columns), rf]).fit(train_df)


def fit_rf_tuned(
    train_df: DataFrame,
    evaluator: BinaryClassificationEvaluator,
    feature_columns: list[str],
    args: argparse.Namespace,
):
    rf = RandomForestClassifier(
        labelCol="fire_occurred",
        featuresCol="features",
        weightCol="class_weight",
        seed=args.seed,
    )
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, args.rf_num_trees_grid)
        .addGrid(rf.maxDepth, args.rf_max_depth_grid)
        .addGrid(rf.minInstancesPerNode, args.rf_min_instances_per_node_grid)
        .addGrid(rf.featureSubsetStrategy, args.rf_feature_subset_strategy_grid)
        .build()
    )
    cv = CrossValidator(
        estimator=Pipeline(stages=[make_assembler(feature_columns), rf]),
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=args.rf_cv_folds,
        seed=args.seed,
        parallelism=args.cv_parallelism,
    )
    return cv.fit(train_df), param_grid


def fit_gbt_tuned(
    train_df: DataFrame,
    evaluator: BinaryClassificationEvaluator,
    feature_columns: list[str],
    args: argparse.Namespace,
):
    gbt = GBTClassifier(
        labelCol="fire_occurred",
        featuresCol="features",
        weightCol="class_weight",
        seed=args.seed,
    )
    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, args.gbt_max_depth_grid)
        .addGrid(gbt.maxIter, args.gbt_max_iter_grid)
        .addGrid(gbt.stepSize, args.gbt_step_size_grid)
        .build()
    )
    cv = CrossValidator(
        estimator=Pipeline(stages=[make_assembler(feature_columns), gbt]),
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=args.gbt_cv_folds,
        seed=args.seed,
        parallelism=args.cv_parallelism,
    )
    return cv.fit(train_df), param_grid


def stage_params(stage: Any) -> dict[str, Any]:
    params = {}
    for param_name in [
        "maxDepth",
        "minInstancesPerNode",
        "featureSubsetStrategy",
        "maxIter",
        "stepSize",
    ]:
        getter = f"get{param_name[0].upper()}{param_name[1:]}"
        if hasattr(stage, getter):
            value = getattr(stage, getter)()
            params[param_name] = value
    if hasattr(stage, "getNumTrees"):
        value = getattr(stage, "getNumTrees")
        params["numTrees"] = value() if callable(value) else value
    return params


def evaluate_model(
    model_name: str,
    model,
    validation_df: DataFrame,
    test_df: DataFrame,
    evaluator_roc: BinaryClassificationEvaluator,
    evaluator_pr: BinaryClassificationEvaluator,
    thresholds: list[float],
    precision_at_k_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validation_predictions = model.transform(validation_df).cache()
    test_predictions = model.transform(test_df).cache()

    validation_scores = collect_scores(validation_predictions)
    test_scores = collect_scores(test_predictions)
    threshold_rows = threshold_curve(validation_scores, thresholds)
    best_threshold, validation_best = choose_best_threshold(threshold_rows)

    test_metrics = binary_metrics_from_scores(test_scores, best_threshold)
    k_metrics = precision_recall_at_k(test_scores, precision_at_k_fraction)
    calibration_rows = calibration_bins(test_scores)
    validation_auc_roc = float(evaluator_roc.evaluate(validation_predictions))
    validation_auc_pr = float(evaluator_pr.evaluate(validation_predictions))
    auc_roc = float(evaluator_roc.evaluate(test_predictions))
    auc_pr = float(evaluator_pr.evaluate(test_predictions))

    metrics = {
        "model": model_name,
        "validation_auc_roc": validation_auc_roc,
        "validation_auc_pr": validation_auc_pr,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "optimal_threshold": best_threshold,
        "validation_best_f1": float(validation_best.get("f1", 0.0)),
        "validation_best_precision": float(validation_best.get("precision", 0.0)),
        "validation_best_recall": float(validation_best.get("recall", 0.0)),
        **test_metrics,
        **k_metrics,
    }

    for row in threshold_rows:
        row["model"] = model_name
    for row in calibration_rows:
        row["model"] = model_name

    validation_predictions.unpersist()
    test_predictions.unpersist()
    return metrics, threshold_rows, calibration_rows, test_scores_to_rows(model_name, test_scores)


def test_scores_to_rows(model_name: str, scores: list[tuple[int, float]]) -> list[dict[str, Any]]:
    return [
        {"model": model_name, "label": label, "positive_probability": probability}
        for label, probability in scores
    ]


def prepare_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    prepare_output_path(path)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_metrics(path: Path, metrics: dict[str, object]) -> None:
    prepare_output_path(path)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_feature_importance_csv(path: Path, feature_columns: list[str], importances: list[float]) -> list[dict[str, Any]]:
    rows = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in sorted(zip(feature_columns, importances), key=lambda item: item[1], reverse=True)
    ]
    write_csv(path, rows, ["feature", "importance"])
    return rows


def plot_feature_importance(path: Path, rows: list[dict[str, Any]], title: str, top_n: int = 15) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_rows = rows[:top_n]
    labels = [str(row["feature"]) for row in reversed(top_rows)]
    values = [float(row["importance"]) for row in reversed(top_rows)]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(labels)), values, color="#2f6f73")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Feature importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_calibration(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    model_names = sorted({str(row["model"]) for row in rows})
    for model_name in model_names:
        model_rows = [row for row in rows if row["model"] == model_name]
        x_values = [float(row["mean_predicted_probability"]) for row in model_rows]
        y_values = [float(row["actual_fire_rate"]) for row in model_rows]
        plt.plot(x_values, y_values, marker="o", linewidth=1.8, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Actual fire rate")
    plt.title("Calibration Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_thresholds(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    model_names = sorted({str(row["model"]) for row in rows})
    for model_name in model_names:
        model_rows = [row for row in rows if row["model"] == model_name]
        x_values = [float(row["threshold"]) for row in model_rows]
        y_values = [float(row["f1"]) for row in model_rows]
        plt.plot(x_values, y_values, marker="o", linewidth=1.8, label=model_name)
    plt.xlabel("Threshold")
    plt.ylabel("Validation F1")
    plt.title("Threshold Optimization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def maybe_setup_mlflow(args: argparse.Namespace):
    if args.skip_mlflow:
        return None
    try:
        import mlflow
        import mlflow.spark  # noqa: F401
    except ImportError as exc:
        if args.require_mlflow:
            raise RuntimeError("MLflow is required but not installed. Install mlflow or pass --skip-mlflow.") from exc
        print("MLflow is not installed; skipping MLflow logging.")
        return None

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_registry_uri(args.mlflow_registry_uri or args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)
    return mlflow


def param_map_to_dict(param_map) -> dict[str, Any]:
    return {param.name: value for param, value in param_map.items()}


def mlflow_metrics_with_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    if "auc_roc" in metrics:
        result["AUC"] = metrics["auc_roc"]
    if "auc_pr" in metrics:
        result["AUC_PR"] = metrics["auc_pr"]
    if "validation_auc_roc" in metrics:
        result["validation_AUC"] = metrics["validation_auc_roc"]
    if "validation_auc_pr" in metrics:
        result["validation_AUC_PR"] = metrics["validation_auc_pr"]
    return result


def find_registered_version(mlflow_module, registered_model_name: str, run_id: str) -> str | None:
    client = mlflow_module.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    matching = [version for version in versions if getattr(version, "run_id", None) == run_id]
    if not matching:
        return None
    return str(max(matching, key=lambda version: int(version.version)).version)


def promote_model_version(mlflow_module, registered_model_name: str, version: str) -> None:
    client = mlflow_module.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(registered_model_name, version, "deployment_stage", "Production")
    client.set_registered_model_tag(registered_model_name, "production_version", str(version))
    try:
        client.set_registered_model_alias(registered_model_name, "production", version)
    except Exception:
        pass


def tag_registered_model_source(
    mlflow_module,
    registered_model_name: str,
    version: str,
    spark_pipeline_uri: str | None,
) -> None:
    if not spark_pipeline_uri:
        return

    client = mlflow_module.tracking.MlflowClient()
    client.set_model_version_tag(registered_model_name, version, "spark_pipeline_uri", spark_pipeline_uri)

def register_spark_pipeline_model(
    mlflow_module,
    registered_model_name: str,
    run_id: str,
    spark_pipeline_uri: str,
) -> str:
    client = mlflow_module.tracking.MlflowClient()
    try:
        client.create_registered_model(registered_model_name)
    except Exception:
        pass
    version = client.create_model_version(
        name=registered_model_name,
        source=spark_pipeline_uri.rstrip("/") + "/",
        run_id=run_id,
    )
    return str(version.version)

def log_cv_candidate_runs(
    mlflow_module,
    model_name: str,
    param_maps: list[Any],
    avg_metrics: list[float],
    cv_folds: int,
    feature_columns: list[str],
) -> None:
    if mlflow_module is None:
        return

    for index, (param_map, avg_metric) in enumerate(zip(param_maps, avg_metrics), start=1):
        log_mlflow_run(
            mlflow_module,
            f"{model_name}-candidate-{index:02d}",
            {
                "model_family": model_name,
                "candidate_index": index,
                "cv_folds": cv_folds,
                "feature_count": len(feature_columns),
                **param_map_to_dict(param_map),
                "_tags": {
                    "run_role": "cv_candidate",
                    "model_family": model_name,
                },
            },
            {
                "cv_avg_auc_roc": float(avg_metric),
                "AUC": float(avg_metric),
            },
            [],
        )


def log_mlflow_run(
    mlflow_module,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifact_paths: list[Path],
) -> dict[str, Any]:
    if mlflow_module is None:
        return {}

    with mlflow_module.start_run(run_name=model_name):
        run_id = mlflow_module.active_run().info.run_id
        tags = params.pop("_tags", None)
        registered_model_name = params.pop("_registered_model_name", None)
        model = params.pop("_model", None)
        dfs_tmpdir = params.pop("_dfs_tmpdir", None)
        promote_to_production = bool(params.pop("_promote_to_production", False))
        spark_pipeline_uri = params.get("model_output")
        if isinstance(tags, dict):
            mlflow_module.set_tags(tags)
        for key, value in params.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, sort_keys=True)
            mlflow_module.log_param(key, value)
        for key, value in mlflow_metrics_with_aliases(metrics).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                mlflow_module.log_metric(key, float(value))
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                mlflow_module.log_artifact(str(artifact_path))
        model_info = None
        registered_version = None
        if model is not None and registered_model_name:
            try:
                model_info = mlflow_module.spark.log_model(
                    spark_model=model,
                    artifact_path="model",
                    registered_model_name=registered_model_name,
                    dfs_tmpdir=dfs_tmpdir,
                )
                registered_version = getattr(model_info, "registered_model_version", None)
                if registered_version is None:
                    registered_version = find_registered_version(mlflow_module, registered_model_name, run_id)
            except Exception as exc:
                if not isinstance(spark_pipeline_uri, str) or not spark_pipeline_uri:
                    raise
                print(
                    "mlflow.spark.log_model failed; registering existing Spark PipelineModel "
                    f"from {spark_pipeline_uri} instead. Original error: {exc}"
                )
                registered_version = register_spark_pipeline_model(
                    mlflow_module,
                    registered_model_name,
                    run_id,
                    spark_pipeline_uri,
                )
            if registered_version:
                tag_registered_model_source(
                    mlflow_module,
                    registered_model_name,
                    str(registered_version),
                    spark_pipeline_uri if isinstance(spark_pipeline_uri, str) else None,
                )
                if promote_to_production:
                    promote_model_version(mlflow_module, registered_model_name, str(registered_version))

    result = {"mlflow_run_id": run_id}
    if registered_model_name:
        result["registered_model_name"] = registered_model_name
    if registered_version:
        result["registered_model_version"] = str(registered_version)
        result["registered_model_uri"] = f"models:/{registered_model_name}/{registered_version}"
        if promote_to_production:
            result["registered_model_stage"] = "Production"
            result["production_model_uri"] = f"models:/{registered_model_name}/Production"
    if model_info is not None:
        result["mlflow_model_uri"] = f"runs:/{run_id}/model"
    return result


def log_artifact_to_mlflow_run(mlflow_module, run_id: str | None, artifact_path: Path) -> None:
    if mlflow_module is None or not run_id or not artifact_path.exists():
        return
    with mlflow_module.start_run(run_id=run_id):
        mlflow_module.log_artifact(str(artifact_path))


def comparison_row(metrics: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "model",
        "validation_auc_roc",
        "validation_auc_pr",
        "auc_roc",
        "auc_pr",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "specificity",
        "precision_at_k",
        "recall_at_k",
        "k",
        "optimal_threshold",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]
    return {field: metrics.get(field) for field in fields}


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(description="Train advanced Spark MLlib wildfire models.")
    parser.add_argument("--spark-app-name", default="Wildfire Advanced ML")
    parser.add_argument("--spark-master", default=os.getenv("SPARK_MASTER"))
    parser.add_argument("--hadoop-aws-package", default=os.getenv("SPARK_HADOOP_AWS_PACKAGE", DEFAULT_HADOOP_AWS_PACKAGE))
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--features-prefix", default=os.getenv("FEATURES_PREFIX", "features"))
    parser.add_argument("--rf-baseline-model-output-prefix", default=os.getenv("RF_BASELINE_MODEL_OUTPUT_PREFIX", "models/rf_baseline"))
    parser.add_argument("--rf-tuned-model-output-prefix", default=os.getenv("RF_TUNED_MODEL_OUTPUT_PREFIX", "models/rf_tuned"))
    parser.add_argument("--gbt-model-output-prefix", default=os.getenv("GBT_MODEL_OUTPUT_PREFIX", "models/gbt"))
    parser.add_argument("--checkpoint-prefix", default=os.getenv("ADVANCED_ML_CHECKPOINT_PREFIX", "checkpoints/advanced_ml"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/model_metrics_week1.json"))
    parser.add_argument("--comparison-output", type=Path, default=Path("reports/model_comparison_week1.csv"))
    parser.add_argument("--importance-output", type=Path, default=Path("reports/feature_importance_week1.csv"))
    parser.add_argument("--importance-plot-output", type=Path, default=Path("reports/feature_importance_week1.png"))
    parser.add_argument("--calibration-output", type=Path, default=Path("reports/calibration_curves_week1.csv"))
    parser.add_argument("--calibration-plot-output", type=Path, default=Path("reports/calibration_curves_week1.png"))
    parser.add_argument("--threshold-output", type=Path, default=Path("reports/threshold_optimization_week1.csv"))
    parser.add_argument("--threshold-plot-output", type=Path, default=Path("reports/threshold_optimization_week1.png"))
    parser.add_argument("--train-end-date", default="2022-12-31")
    parser.add_argument("--validation-start-date", default="2023-01-01")
    parser.add_argument("--validation-end-date", default="2023-12-31")
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--rf-baseline-num-trees", type=int, default=50)
    parser.add_argument("--rf-baseline-max-depth", type=int, default=5)
    parser.add_argument("--rf-baseline-min-instances-per-node", type=int, default=5)
    parser.add_argument("--rf-baseline-feature-subset-strategy", default="sqrt")
    parser.add_argument("--rf-cv-folds", type=int, default=3)
    parser.add_argument("--rf-num-trees-grid", default="80,100")
    parser.add_argument("--rf-max-depth-grid", default="8,10")
    parser.add_argument("--rf-min-instances-per-node-grid", default="1")
    parser.add_argument("--rf-feature-subset-strategy-grid", default="sqrt")
    parser.add_argument("--gbt-cv-folds", type=int, default=2)
    parser.add_argument("--gbt-max-depth-grid", default="3,5")
    parser.add_argument("--gbt-max-iter-grid", default="80")
    parser.add_argument("--gbt-step-size-grid", default="0.1")
    parser.add_argument("--cv-parallelism", type=int, default=1)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    parser.add_argument("--driver-memory", default=os.getenv("SPARK_DRIVER_MEMORY", "4g"))
    parser.add_argument("--executor-memory", default=os.getenv("SPARK_EXECUTOR_MEMORY", "4g"))
    parser.add_argument("--spark-log-level", default="WARN")
    parser.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--precision-at-k-fraction", type=float, default=0.10)
    parser.add_argument("--skip-gbt", action="store_true")
    parser.add_argument("--skip-mlflow", action="store_true")
    parser.add_argument("--skip-model-registry", action="store_true")
    parser.add_argument("--require-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--mlflow-registry-uri", default=os.getenv("MLFLOW_REGISTRY_URI"))
    parser.add_argument("--mlflow-experiment-name", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "wildfire-prediction"))
    parser.add_argument("--rf-tuned-registered-model-name", default=os.getenv("RF_TUNED_REGISTERED_MODEL_NAME", REGISTERED_MODEL_NAMES["rf_tuned"]))
    parser.add_argument("--gbt-registered-model-name", default=os.getenv("GBT_REGISTERED_MODEL_NAME", REGISTERED_MODEL_NAMES["gbt"]))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timezone", default=os.getenv("SPARK_SQL_TIMEZONE", "UTC"))

    args = parser.parse_args()
    args.rf_num_trees_grid = parse_int_list(args.rf_num_trees_grid)
    args.rf_max_depth_grid = parse_int_list(args.rf_max_depth_grid)
    args.rf_min_instances_per_node_grid = parse_int_list(args.rf_min_instances_per_node_grid)
    args.rf_feature_subset_strategy_grid = parse_str_list(args.rf_feature_subset_strategy_grid)
    args.gbt_max_depth_grid = parse_int_list(args.gbt_max_depth_grid)
    args.gbt_max_iter_grid = parse_int_list(args.gbt_max_iter_grid)
    args.gbt_step_size_grid = parse_float_list(args.gbt_step_size_grid)
    args.thresholds = parse_float_list(args.thresholds)

    if args.rf_cv_folds < 2:
        parser.error("--rf-cv-folds must be at least 2")
    if args.gbt_cv_folds < 2:
        parser.error("--gbt-cv-folds must be at least 2")
    if args.cv_parallelism <= 0:
        parser.error("--cv-parallelism must be positive")
    if args.shuffle_partitions <= 0:
        parser.error("--shuffle-partitions must be positive")
    if not 0 < args.precision_at_k_fraction <= 1:
        parser.error("--precision-at-k-fraction must be in (0, 1]")
    for grid_name in [
        "rf_num_trees_grid",
        "rf_max_depth_grid",
        "rf_min_instances_per_node_grid",
        "rf_feature_subset_strategy_grid",
        "gbt_max_depth_grid",
        "gbt_max_iter_grid",
        "gbt_step_size_grid",
        "thresholds",
    ]:
        if not getattr(args, grid_name):
            parser.error(f"--{grid_name.replace('_', '-')} must not be empty")
    return args


def main() -> int:
    args = parse_args()
    features_input = s3a_path(args.minio_bucket, args.features_prefix)
    rf_baseline_output = s3a_path(args.minio_bucket, args.rf_baseline_model_output_prefix)
    rf_tuned_output = s3a_path(args.minio_bucket, args.rf_tuned_model_output_prefix)
    gbt_output = s3a_path(args.minio_bucket, args.gbt_model_output_prefix)
    checkpoint_output = s3a_path(args.minio_bucket, args.checkpoint_prefix)
    mlflow_dfs_tmp = s3a_path(args.minio_bucket, f"{args.checkpoint_prefix.strip('/')}/mlflow_tmp")

    spark = build_spark(args)
    spark.sparkContext.setLogLevel(args.spark_log_level)
    spark.sparkContext.setCheckpointDir(checkpoint_output)
    mlflow_module = maybe_setup_mlflow(args)

    try:
        data = spark.read.parquet(features_input)
        data = (
            prepare_data(data, DEFAULT_FEATURE_COLUMNS)
            .repartition(args.shuffle_partitions, "grid_id")
            .checkpoint(eager=True)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        train_base = data.filter(F.col("date") <= F.to_date(F.lit(args.train_end_date))).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        validation_df = data.filter(
            (F.col("date") >= F.to_date(F.lit(args.validation_start_date)))
            & (F.col("date") <= F.to_date(F.lit(args.validation_end_date)))
        ).persist(StorageLevel.MEMORY_AND_DISK)
        test_df = data.filter(F.col("date") >= F.to_date(F.lit(args.test_start_date))).persist(
            StorageLevel.MEMORY_AND_DISK
        )

        train_df, weight_info = add_class_weights(train_base)
        train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)

        train_rows = train_df.count()
        validation_rows = validation_df.count()
        test_rows = test_df.count()
        data_bounds = data.agg(
            F.min("date").alias("date_min"),
            F.max("date").alias("date_max"),
            F.countDistinct("grid_id").alias("grid_count"),
        ).collect()[0]

        evaluator_roc = BinaryClassificationEvaluator(
            labelCol="fire_occurred",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )
        evaluator_pr = BinaryClassificationEvaluator(
            labelCol="fire_occurred",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR",
        )

        model_metrics: dict[str, Any] = {
            "features_input": features_input,
            "feature_columns": DEFAULT_FEATURE_COLUMNS,
            "feature_dataset": {
                "date_min": data_bounds["date_min"].isoformat() if data_bounds["date_min"] else None,
                "date_max": data_bounds["date_max"].isoformat() if data_bounds["date_max"] else None,
                "grid_count": int(data_bounds["grid_count"] or 0),
            },
            "class_weights": weight_info,
            "splits": {
                "train_end_date": args.train_end_date,
                "validation_start_date": args.validation_start_date,
                "validation_end_date": args.validation_end_date,
                "test_start_date": args.test_start_date,
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "test_rows": test_rows,
            },
        }
        comparison_rows: list[dict[str, Any]] = []
        threshold_rows: list[dict[str, Any]] = []
        calibration_rows: list[dict[str, Any]] = []
        score_rows: list[dict[str, Any]] = []

        print("Training RF baseline...")
        rf_baseline_model = fit_rf_baseline(train_df, DEFAULT_FEATURE_COLUMNS, args)
        rf_baseline_tree = extract_stage_model(rf_baseline_model, RandomForestClassificationModel)
        rf_baseline_metrics, rf_thresholds, rf_calibration, rf_scores = evaluate_model(
            "rf_baseline",
            rf_baseline_model,
            validation_df,
            test_df,
            evaluator_roc,
            evaluator_pr,
            args.thresholds,
            args.precision_at_k_fraction,
        )
        rf_baseline_metrics.update(
            {
                "model_type": "RandomForestClassifier",
                "model_output": rf_baseline_output,
                "params": stage_params(rf_baseline_tree),
            }
        )
        rf_baseline_model.write().overwrite().save(rf_baseline_output)
        model_metrics["rf_baseline"] = rf_baseline_metrics
        comparison_rows.append(comparison_row(rf_baseline_metrics))
        threshold_rows.extend(rf_thresholds)
        calibration_rows.extend(rf_calibration)
        score_rows.extend(rf_scores)

        print("Training tuned RF with CrossValidator...")
        rf_cv_model, rf_param_grid = fit_rf_tuned(train_df, evaluator_roc, DEFAULT_FEATURE_COLUMNS, args)
        rf_tuned_model = rf_cv_model.bestModel
        rf_tuned_tree = extract_stage_model(rf_tuned_model, RandomForestClassificationModel)
        rf_tuned_metrics, tuned_thresholds, tuned_calibration, tuned_scores = evaluate_model(
            "rf_tuned",
            rf_tuned_model,
            validation_df,
            test_df,
            evaluator_roc,
            evaluator_pr,
            args.thresholds,
            args.precision_at_k_fraction,
        )
        rf_tuned_metrics.update(
            {
                "model_type": "RandomForestClassifier",
                "model_output": rf_tuned_output,
                "params": stage_params(rf_tuned_tree),
                "cv_folds": args.rf_cv_folds,
                "cv_grid_size": len(rf_param_grid),
                "cv_avg_metrics": [float(value) for value in rf_cv_model.avgMetrics],
            }
        )
        rf_tuned_model.write().overwrite().save(rf_tuned_output)
        model_metrics["rf_tuned"] = rf_tuned_metrics
        comparison_rows.append(comparison_row(rf_tuned_metrics))
        threshold_rows.extend(tuned_thresholds)
        calibration_rows.extend(tuned_calibration)
        score_rows.extend(tuned_scores)

        gbt_metrics = None
        if not args.skip_gbt:
            print("Training tuned GBT with CrossValidator...")
            gbt_cv_model, gbt_param_grid = fit_gbt_tuned(train_df, evaluator_roc, DEFAULT_FEATURE_COLUMNS, args)
            gbt_model = gbt_cv_model.bestModel
            gbt_tree = extract_stage_model(gbt_model, GBTClassificationModel)
            gbt_metrics, gbt_thresholds, gbt_calibration, gbt_scores = evaluate_model(
                "gbt",
                gbt_model,
                validation_df,
                test_df,
                evaluator_roc,
                evaluator_pr,
                args.thresholds,
                args.precision_at_k_fraction,
            )
            gbt_metrics.update(
                {
                    "model_type": "GBTClassifier",
                    "model_output": gbt_output,
                    "params": stage_params(gbt_tree),
                    "cv_folds": args.gbt_cv_folds,
                    "cv_grid_size": len(gbt_param_grid),
                    "cv_avg_metrics": [float(value) for value in gbt_cv_model.avgMetrics],
                }
            )
            gbt_model.write().overwrite().save(gbt_output)
            model_metrics["gbt"] = gbt_metrics
            comparison_rows.append(comparison_row(gbt_metrics))
            threshold_rows.extend(gbt_thresholds)
            calibration_rows.extend(gbt_calibration)
            score_rows.extend(gbt_scores)

        best_row = max(comparison_rows, key=lambda row: float(row["validation_auc_roc"]))
        rf_auc_uplift = float(rf_tuned_metrics["auc_roc"]) - float(rf_baseline_metrics["auc_roc"])
        gbt_auc_delta = (
            float(gbt_metrics["auc_roc"]) - float(rf_tuned_metrics["auc_roc"])
            if gbt_metrics is not None
            else None
        )
        model_metrics["best_model"] = best_row["model"]
        model_metrics["best_model_selection"] = {
            "metric": "validation_auc_roc",
            "source": "validation_split",
            "value": float(best_row["validation_auc_roc"]),
        }
        model_metrics["rf_tuned_auc_uplift"] = rf_auc_uplift
        model_metrics["rf_tuned_auc_target_met"] = rf_auc_uplift >= 0.03
        model_metrics["gbt_auc_delta_vs_rf_tuned"] = gbt_auc_delta
        model_metrics["gbt_auc_target_met"] = bool(gbt_auc_delta is not None and gbt_auc_delta >= 0.0)

        importance_rows = write_feature_importance_csv(
            args.importance_output,
            DEFAULT_FEATURE_COLUMNS,
            rf_tuned_tree.featureImportances.toArray().tolist(),
        )
        write_csv(args.comparison_output, comparison_rows)
        write_csv(args.threshold_output, threshold_rows)
        write_csv(args.calibration_output, calibration_rows)
        write_metrics(args.metrics_output, model_metrics)
        plot_feature_importance(args.importance_plot_output, importance_rows, "Top Feature Importance - RF Tuned")
        plot_thresholds(args.threshold_plot_output, threshold_rows)
        plot_calibration(args.calibration_plot_output, calibration_rows)

        common_artifacts = [
            args.metrics_output,
            args.comparison_output,
            args.importance_output,
            args.importance_plot_output,
            args.threshold_output,
            args.threshold_plot_output,
            args.calibration_output,
            args.calibration_plot_output,
        ]
        log_cv_candidate_runs(
            mlflow_module,
            "rf_tuned",
            rf_param_grid,
            [float(value) for value in rf_cv_model.avgMetrics],
            args.rf_cv_folds,
            DEFAULT_FEATURE_COLUMNS,
        )
        if gbt_metrics is not None:
            log_cv_candidate_runs(
                mlflow_module,
                "gbt",
                gbt_param_grid,
                [float(value) for value in gbt_cv_model.avgMetrics],
                args.gbt_cv_folds,
                DEFAULT_FEATURE_COLUMNS,
            )

        log_mlflow_run(
            mlflow_module,
            "rf_baseline",
            {
                "model_output": rf_baseline_output,
                "feature_count": len(DEFAULT_FEATURE_COLUMNS),
                "feature_columns": DEFAULT_FEATURE_COLUMNS,
                "_tags": {"run_role": "final_model", "model_family": "rf_baseline"},
                **rf_baseline_metrics["params"],
            },
            rf_baseline_metrics,
            common_artifacts,
        )
        rf_registry_info = log_mlflow_run(
            mlflow_module,
            "rf_tuned",
            {
                "model_output": rf_tuned_output,
                "feature_count": len(DEFAULT_FEATURE_COLUMNS),
                "feature_columns": DEFAULT_FEATURE_COLUMNS,
                "_tags": {"run_role": "final_model", "model_family": "rf_tuned"},
                "_model": rf_tuned_model,
                "_dfs_tmpdir": mlflow_dfs_tmp,
                "_registered_model_name": None if args.skip_model_registry else args.rf_tuned_registered_model_name,
                "_promote_to_production": best_row["model"] == "rf_tuned",
                **rf_tuned_metrics["params"],
            },
            rf_tuned_metrics,
            common_artifacts,
        )
        if rf_registry_info:
            model_metrics["rf_tuned"].update(rf_registry_info)
        gbt_registry_info: dict[str, Any] = {}
        if gbt_metrics is not None:
            gbt_registry_info = log_mlflow_run(
                mlflow_module,
                "gbt",
                {
                    "model_output": gbt_output,
                    "feature_count": len(DEFAULT_FEATURE_COLUMNS),
                    "feature_columns": DEFAULT_FEATURE_COLUMNS,
                    "_tags": {"run_role": "final_model", "model_family": "gbt"},
                    "_model": gbt_model,
                    "_dfs_tmpdir": mlflow_dfs_tmp,
                    "_registered_model_name": None if args.skip_model_registry else args.gbt_registered_model_name,
                    "_promote_to_production": best_row["model"] == "gbt",
                    **gbt_metrics["params"],
                },
                gbt_metrics,
                common_artifacts,
            )
            if gbt_registry_info:
                model_metrics["gbt"].update(gbt_registry_info)

        write_metrics(args.metrics_output, model_metrics)
        for registry_info in (rf_registry_info, gbt_registry_info):
            run_id = registry_info.get("mlflow_run_id") if registry_info else None
            log_artifact_to_mlflow_run(
                mlflow_module,
                run_id if isinstance(run_id, str) else None,
                args.metrics_output,
            )

        print("Model comparison:")
        for row in comparison_rows:
            print(
                f"{row['model']}: AUC-ROC={float(row['auc_roc']):.4f}, "
                f"AUC-PR={float(row['auc_pr']):.4f}, F1={float(row['f1']):.4f}, "
                f"Precision@K={float(row['precision_at_k']):.4f}, Recall={float(row['recall']):.4f}"
            )
        print(f"RF tuned AUC uplift vs baseline: {rf_auc_uplift:.4f}")
        if gbt_auc_delta is not None:
            print(f"GBT AUC delta vs RF tuned: {gbt_auc_delta:.4f}")
        print(f"Best model by validation AUC-ROC: {best_row['model']}")
        print(f"Saved metrics to {args.metrics_output}")
        print(f"Saved comparison table to {args.comparison_output}")
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
