FROM apache/spark:3.5.6

USER root

RUN python3 -m pip install --no-cache-dir \
    numpy==1.24.4 \
    pandas==2.0.3 \
    packaging==24.2 \
    matplotlib==3.7.5 \
    mlflow==2.17.2 \
    apache-sedona==1.5.1 \
    shapely==2.0.7

USER spark
