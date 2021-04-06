#!/bin/bash

sbt compile && spark-submit \
  --class org.sustain.Main  \
  --master spark://lattice  \
  target/scala-2.12/transfer-learning_2.12-1.0.jar