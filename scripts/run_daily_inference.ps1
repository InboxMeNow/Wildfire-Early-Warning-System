param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

docker compose up -d
docker compose exec -T spark-master /opt/spark/bin/spark-submit `
    --master spark://spark-master:7077 `
    /workspace/09_inference.py
