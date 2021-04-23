#!/bin/bash

spark-shell \
  --master spark://lattice-100:8079  \
  --name "transfer-learning-shell-experiments" \
  --total-executor-cores 4 \
  --num-executors 4 \
  --executor-memory 4G \
  --conf spark.executorEnv.JAVA_HOME=$JAVA_HOME \
  --conf spark.mongodb.input.uri=mongodb://lattice-100:27018/ \
  --conf spark.mongodb.input.database=sustaindb \
  --conf spark.mongodb.input.collection=noaa_nam \
  --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1