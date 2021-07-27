package org.sustain

import org.apache.spark.sql.SparkSession
import org.sustain.util.Logger

object SparkManager {

  def logEnv(): Unit = {
    Logger.log(">>> Log Environment")
    Logger.log("USE_KUBERNETES: " + Constants.USE_KUBERNETES)
    if (Constants.USE_KUBERNETES) {
      Logger.log("SPARK_MASTER: " + Constants.KUBERNETES_SPARK_MASTER)
    } else {
      Logger.log("SPARK_MASTER: " + Constants.SPARK_MASTER)
    }
    Logger.log("DB_HOST: " + Constants.DB_HOST)
    Logger.log("DB_PORT: " + Constants.DB_PORT)
  }

  def getSparkMaster(): String = {
    if (Constants.USE_KUBERNETES) {
      Constants.KUBERNETES_SPARK_MASTER
    } else {
      Constants.SPARK_MASTER
    }
  }

  def getSparkSession(collection1: String): SparkSession = {
    if (Constants.USE_KUBERNETES) {
      SparkSession.builder()
        .master(Constants.KUBERNETES_SPARK_MASTER)
        .appName(s"Clustering ('$collection1'): Varying #clusters")
        .config("spark.submit.deployMode", "cluster")
        .config("spark.mongodb.input.uri",
          "mongodb://" + Constants.DB_HOST + ":" + Constants.DB_PORT + "/sustaindb." + collection1)
        .config("spark.kubernetes.container.image", Constants.SPARK_DOCKER_IMAGE)
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.shuffleTracking.enabled", "true")
        //        .config("spark.executor.instances", Constants.SPARK_INITIAL_EXECUTORS)
        .config("spark.dynamicAllocation.minExecutors", Constants.SPARK_INITIAL_EXECUTORS)
        .config("spark.dynamicAllocation.maxExecutors", Constants.SPARK_MAX_EXECUTORS)
        .config("spark.executor.memory", Constants.SPARK_EXECUTOR_MEMORY)
        .getOrCreate()
    } else {
      SparkSession.builder()
        .master(Constants.SPARK_MASTER)
        .appName(s"Clustering ('$collection1'): Varying #clusters")
        .config("spark.submit.deployMode", "cluster")
        .config("spark.mongodb.input.uri",
          "mongodb://" + Constants.DB_HOST + ":" + Constants.DB_PORT + "/sustaindb." + collection1)
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.shuffleTracking.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", Constants.SPARK_INITIAL_EXECUTORS)
        .config("spark.dynamicAllocation.maxExecutors", Constants.SPARK_MAX_EXECUTORS)
        .config("spark.executor.memory", Constants.SPARK_EXECUTOR_MEMORY)
        .getOrCreate()
    }
  }
}
