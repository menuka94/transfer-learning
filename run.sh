#!/bin/bash

sbt compile && sbt package && spark-submit \
  --class org.sustain.Main  \
  --conf spark.executorEnv.JAVA_HOME=$JAVA_HOME \
  --executor-memory 20G \
  --total-executor-cores 100 \
  --master spark://lattice-100:8079  \
  --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
  target/scala-2.12/transfer-learning_2.12-1.0.jar
