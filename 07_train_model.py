#!/usr/bin/env python3
"""
Train Spark MLlib wildfire occurrence models.

Input:
    s3a://wildfire-data/features/

Output:
    s3a://wildfire-data/models/random_forest_fire_baseline/
    s3a://wildfire-data/models/gbt_fire_baseline/
    reports/model_metrics_week1.json
    reports/feature_importance_week1.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassificationModel,
    GBTClassifier,
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
DEFAULT_MINIO_ACCESS_KEY = "minioadmin"
DEFAULT_MINIO_SECRET_KEY = "minioadmin"
DEFAULT_MINIO_BUCKET = "wildfire-data"
DEFAULT_HADOOP_AWS_PACKAGE = "org.apache.hadoop:hadoop-aws:3.3.4"
DEFAULT_FEATURE_COLUMNS = [
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
    "station_distance_km",
    "weather_points_count",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]


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


def confusion_metrics(predictions: DataFrame) -> dict[str, float | int]:
    matrix = {
        (int(row["fire_occurred"]), int(row["prediction"])): int(row["count"])
        for row in predictions.groupBy("fire_occurred", "prediction").count().collect()
    }
    tn = matrix.get((0, 0), 0)
    fp = matrix.get((0, 1), 0)
    fn = matrix.get((1, 0), 0)
    tp = matrix.get((1, 1), 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0

    return {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def extract_stage_model(model, model_type):
    for stage in model.stages:
        if isinstance(stage, model_type):
            return stage
    raise RuntimeError(f"PipelineModel does not contain {model_type.__name__}")


def prepare_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def write_feature_importance(path: Path, feature_columns: list[str], importances: list[float]) -> None:
    prepare_output_path(path)
    rows = sorted(zip(feature_columns, importances), key=lambda item: item[1], reverse=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("feature,importance\n")
        for feature, importance in rows:
            file.write(f"{feature},{importance:.12f}\n")


def write_metrics(path: Path, metrics: dict[str, object]) -> None:
    prepare_output_path(path)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_predictions(
    predictions: DataFrame,
    evaluator: BinaryClassificationEvaluator,
) -> dict[str, float | int]:
    auc_roc = float(evaluator.evaluate(predictions))
    metrics = confusion_metrics(predictions)
    metrics["auc_roc"] = auc_roc
    return metrics


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(description="Train Spark MLlib wildfire baseline models.")
    parser.add_argument("--spark-app-name", default="Wildfire ML Baselines")
    parser.add_argument("--spark-master", default=os.getenv("SPARK_MASTER"))
    parser.add_argument("--hadoop-aws-package", default=os.getenv("SPARK_HADOOP_AWS_PACKAGE", DEFAULT_HADOOP_AWS_PACKAGE))
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--features-prefix", default=os.getenv("FEATURES_PREFIX", "features"))
    parser.add_argument("--rf-model-output-prefix", default=os.getenv("RF_MODEL_OUTPUT_PREFIX", "models/random_forest_fire_baseline"))
    parser.add_argument("--gbt-model-output-prefix", default=os.getenv("GBT_MODEL_OUTPUT_PREFIX", "models/gbt_fire_baseline"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/model_metrics_week1.json"))
    parser.add_argument("--importance-output", type=Path, default=Path("reports/feature_importance_week1.csv"))
    parser.add_argument("--train-end-date", default="2023-12-31")
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--num-trees", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--rf-max-depth-grid",
        default="6,8,10",
        help="Comma-separated RF maxDepth values for CrossValidator.",
    )
    parser.add_argument(
        "--rf-num-trees-grid",
        default="100",
        help="Comma-separated RF numTrees values for CrossValidator.",
    )
    parser.add_argument("--gbt-max-iter", type=int, default=80)
    parser.add_argument("--gbt-max-depth", type=int, default=5)
    parser.add_argument(
        "--skip-gbt",
        action="store_true",
        help="Only train/tune RandomForest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timezone", default=os.getenv("SPARK_SQL_TIMEZONE", "UTC"))

    args = parser.parse_args()
    if args.num_trees <= 0:
        parser.error("--num-trees must be positive")
    if args.max_depth <= 0:
        parser.error("--max-depth must be positive")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    args.rf_max_depth_grid = [int(value.strip()) for value in args.rf_max_depth_grid.split(",") if value.strip()]
    args.rf_num_trees_grid = [int(value.strip()) for value in args.rf_num_trees_grid.split(",") if value.strip()]
    if not args.rf_max_depth_grid or not args.rf_num_trees_grid:
        parser.error("RF parameter grids must not be empty")
    return args


def main() -> int:
    args = parse_args()
    features_input = s3a_path(args.minio_bucket, args.features_prefix)
    rf_model_output = s3a_path(args.minio_bucket, args.rf_model_output_prefix)
    gbt_model_output = s3a_path(args.minio_bucket, args.gbt_model_output_prefix)

    spark = build_spark(args)
    try:
        data = spark.read.parquet(features_input)
        data = add_time_features(data)
        data = data.fillna(0.0, subset=DEFAULT_FEATURE_COLUMNS)

        train_df = data.filter(F.col("date") <= F.to_date(F.lit(args.train_end_date)))
        test_df = data.filter(F.col("date") >= F.to_date(F.lit(args.test_start_date)))
        train_df, weight_info = add_class_weights(train_df)

        assembler = VectorAssembler(
            inputCols=DEFAULT_FEATURE_COLUMNS,
            outputCol="features",
            handleInvalid="keep",
        )
        rf = RandomForestClassifier(
            labelCol="fire_occurred",
            featuresCol="features",
            weightCol="class_weight",
            numTrees=args.num_trees,
            maxDepth=args.max_depth,
            seed=args.seed,
        )
        evaluator = BinaryClassificationEvaluator(
            labelCol="fire_occurred",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )

        rf_pipeline = Pipeline(stages=[assembler, rf])
        rf_param_grid = (
            ParamGridBuilder()
            .addGrid(rf.maxDepth, args.rf_max_depth_grid)
            .addGrid(rf.numTrees, args.rf_num_trees_grid)
            .build()
        )
        rf_cv = CrossValidator(
            estimator=rf_pipeline,
            estimatorParamMaps=rf_param_grid,
            evaluator=evaluator,
            numFolds=args.cv_folds,
            seed=args.seed,
            parallelism=2,
        )
        rf_cv_model = rf_cv.fit(train_df)
        rf_model = rf_cv_model.bestModel
        rf_predictions = rf_model.transform(test_df).cache()
        rf_metrics = evaluate_predictions(rf_predictions, evaluator)
        rf_tree_model = extract_stage_model(rf_model, RandomForestClassificationModel)
        rf_metrics.update(
            {
                "model_type": "RandomForestClassifier",
                "train_rows": train_df.count(),
                "test_rows": test_df.count(),
                "train_end_date": args.train_end_date,
                "test_start_date": args.test_start_date,
                "num_trees": rf_tree_model.getNumTrees,
                "max_depth": rf_tree_model.getMaxDepth(),
                "cv_folds": args.cv_folds,
                "cv_avg_metrics": [float(value) for value in rf_cv_model.avgMetrics],
                "rf_max_depth_grid": args.rf_max_depth_grid,
                "rf_num_trees_grid": args.rf_num_trees_grid,
                "feature_columns": DEFAULT_FEATURE_COLUMNS,
                "class_weights": weight_info,
                "features_input": features_input,
                "model_output": rf_model_output,
            }
        )
        rf_model.write().overwrite().save(rf_model_output)

        model_metrics: dict[str, object] = {"random_forest": rf_metrics}
        best_model_name = "random_forest"

        if not args.skip_gbt:
            gbt = GBTClassifier(
                labelCol="fire_occurred",
                featuresCol="features",
                weightCol="class_weight",
                maxIter=args.gbt_max_iter,
                maxDepth=args.gbt_max_depth,
                seed=args.seed,
            )
            gbt_pipeline = Pipeline(stages=[assembler, gbt])
            gbt_model = gbt_pipeline.fit(train_df)
            gbt_predictions = gbt_model.transform(test_df).cache()
            gbt_metrics = evaluate_predictions(gbt_predictions, evaluator)
            gbt_tree_model = extract_stage_model(gbt_model, GBTClassificationModel)
            gbt_metrics.update(
                {
                    "model_type": "GBTClassifier",
                    "train_rows": train_df.count(),
                    "test_rows": test_df.count(),
                    "train_end_date": args.train_end_date,
                    "test_start_date": args.test_start_date,
                    "max_iter": args.gbt_max_iter,
                    "max_depth": gbt_tree_model.getMaxDepth(),
                    "feature_columns": DEFAULT_FEATURE_COLUMNS,
                    "class_weights": weight_info,
                    "features_input": features_input,
                    "model_output": gbt_model_output,
                }
            )
            gbt_model.write().overwrite().save(gbt_model_output)
            model_metrics["gbt"] = gbt_metrics
            if float(gbt_metrics["auc_roc"]) > float(rf_metrics["auc_roc"]):
                best_model_name = "gbt"

        write_feature_importance(
            args.importance_output,
            DEFAULT_FEATURE_COLUMNS,
            rf_tree_model.featureImportances.toArray().tolist(),
        )
        model_metrics["best_model"] = best_model_name
        write_metrics(args.metrics_output, model_metrics)

        print(f"RF AUC-ROC: {rf_metrics['auc_roc']:.4f}")
        print(f"RF Precision: {rf_metrics['precision']:.4f}")
        print(f"RF Recall: {rf_metrics['recall']:.4f}")
        print(f"RF F1: {rf_metrics['f1']:.4f}")
        if "gbt" in model_metrics:
            gbt_metrics = model_metrics["gbt"]
            print(f"GBT AUC-ROC: {gbt_metrics['auc_roc']:.4f}")
            print(f"GBT Precision: {gbt_metrics['precision']:.4f}")
            print(f"GBT Recall: {gbt_metrics['recall']:.4f}")
            print(f"GBT F1: {gbt_metrics['f1']:.4f}")
        print(f"Best model by AUC-ROC: {best_model_name}")
        print(f"Saved RF model to {rf_model_output}")
        if not args.skip_gbt:
            print(f"Saved GBT model to {gbt_model_output}")
        print(f"Saved metrics to {args.metrics_output}")
        print(f"Saved feature importance to {args.importance_output}")
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
