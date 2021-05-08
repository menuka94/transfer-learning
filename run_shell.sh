#!/bin/bash

# --conf spark.executorEnv.JAVA_HOME=$JAVA_HOME \

spark-shell \
  --master spark://lattice-100:8079 \
  --name transfer-learning-shell-experiments \
  --total-executor-cores 100 \
  --num-executors 8 \
  --executor-memory 20G \
  --conf spark.mongodb.input.uri=mongodb://lattice-100:27018/ \
  --conf spark.mongodb.input.database=sustaindb \
  --conf spark.mongodb.input.collection=noaa_nam \
  --jars vectordisassember_2.12-1.0.jar \
  --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1

