#!/usr/bin/env python3
"""
Train a Spark MLlib RandomForest baseline for wildfire occurrence.

Input:
    s3a://wildfire-data/features/

Output:
    s3a://wildfire-data/models/random_forest_fire_baseline/
    reports/model_metrics_week1.json
    reports/feature_importance_week1.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassificationModel, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
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


def extract_rf_model(model) -> RandomForestClassificationModel:
    for stage in model.stages:
        if isinstance(stage, RandomForestClassificationModel):
            return stage
    raise RuntimeError("PipelineModel does not contain RandomForestClassificationModel")


def write_feature_importance(path: Path, feature_columns: list[str], importances: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(zip(feature_columns, importances), key=lambda item: item[1], reverse=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("feature,importance\n")
        for feature, importance in rows:
            file.write(f"{feature},{importance:.12f}\n")


def write_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(description="Train Spark MLlib RandomForest wildfire baseline.")
    parser.add_argument("--spark-app-name", default="Wildfire RF Baseline")
    parser.add_argument("--spark-master", default=os.getenv("SPARK_MASTER"))
    parser.add_argument("--hadoop-aws-package", default=os.getenv("SPARK_HADOOP_AWS_PACKAGE", DEFAULT_HADOOP_AWS_PACKAGE))
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY))
    parser.add_argument("--minio-bucket", default=os.getenv("MINIO_BUCKET", DEFAULT_MINIO_BUCKET))
    parser.add_argument("--features-prefix", default=os.getenv("FEATURES_PREFIX", "features"))
    parser.add_argument("--model-output-prefix", default=os.getenv("MODEL_OUTPUT_PREFIX", "models/random_forest_fire_baseline"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/model_metrics_week1.json"))
    parser.add_argument("--importance-output", type=Path, default=Path("reports/feature_importance_week1.csv"))
    parser.add_argument("--train-end-date", default="2023-12-31")
    parser.add_argument("--test-start-date", default="2024-01-01")
    parser.add_argument("--num-trees", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timezone", default=os.getenv("SPARK_SQL_TIMEZONE", "UTC"))

    args = parser.parse_args()
    if args.num_trees <= 0:
        parser.error("--num-trees must be positive")
    if args.max_depth <= 0:
        parser.error("--max-depth must be positive")
    return args


def main() -> int:
    args = parse_args()
    features_input = s3a_path(args.minio_bucket, args.features_prefix)
    model_output = s3a_path(args.minio_bucket, args.model_output_prefix)

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
        pipeline = Pipeline(stages=[assembler, rf])
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df).cache()

        evaluator = BinaryClassificationEvaluator(
            labelCol="fire_occurred",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )
        auc_roc = float(evaluator.evaluate(predictions))
        metrics = confusion_metrics(predictions)
        metrics.update(
            {
                "auc_roc": auc_roc,
                "train_rows": train_df.count(),
                "test_rows": test_df.count(),
                "train_end_date": args.train_end_date,
                "test_start_date": args.test_start_date,
                "num_trees": args.num_trees,
                "max_depth": args.max_depth,
                "feature_columns": DEFAULT_FEATURE_COLUMNS,
                "class_weights": weight_info,
                "features_input": features_input,
                "model_output": model_output,
            }
        )

        model.write().overwrite().save(model_output)

        rf_model = extract_rf_model(model)
        write_feature_importance(
            args.importance_output,
            DEFAULT_FEATURE_COLUMNS,
            rf_model.featureImportances.toArray().tolist(),
        )
        write_metrics(args.metrics_output, metrics)

        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Saved model to {model_output}")
        print(f"Saved metrics to {args.metrics_output}")
        print(f"Saved feature importance to {args.importance_output}")
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
