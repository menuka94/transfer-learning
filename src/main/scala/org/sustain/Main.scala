/* -----------------------------------------------
 * Main.scala -
 *
 * Description:
 *    Provides a demonstration of the Spark capabilities.
 *    Guide for this project taken directly from MongoDB docs:
 *    https://docs.mongodb.com/spark-connector/master/scala-api
 *
 *  Author:
 *    Caleb Carlson
 *
 * ----------------------------------------------- */
package org.sustain

import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset, Row, RowFactory}
import org.apache.spark.sql.functions.{col, min, row_number}
import org.apache.spark.sql.types.{ArrayType, DataTypes, FloatType}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import com.mongodb.spark._
import com.mongodb.spark.MongoSpark
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Dataset, Row, SparkSession}

import java.util
import java.util.List

object Main {

  /* Global Variables */
  val SPARK_MASTER: String = "spark://lattice-100:8079"
  val APP_NAME: String = "Transfer Learning"
  val MONGO_URI: String = "mongodb://lattice-100:27018/"
  val MONGO_DB: String = "sustaindb"
  val MONGO_COLLECTION: String = "noaa_nam"

  /* Entrypoint for the application */
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf()
      .setMaster(SPARK_MASTER)
      .setAppName(APP_NAME)
      .set("spark.executor.cores", "2")
      .set("spark.executor.memory", "1G")
      .set("spark.mongodb.input.uri", MONGO_URI)
      .set("spark.mongodb.input.database", MONGO_DB)
      .set("spark.mongodb.input.collection", MONGO_COLLECTION)

    // Create the SparkSession and ReadConfig
    val sparkConnector: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate() // For the $()-referenced columns

    val experiment: Experiment = new Experiment(sparkConnector)
    val clusterCenters: Array[String] = time { experiment.cluster() }
    time { experiment.trainCenters(clusterCenters) }

  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("\n\n>>> Elapsed time: " + ( ( (t1 - t0) / 1000) / 1000 )  + " seconds")
    result
  }


}
