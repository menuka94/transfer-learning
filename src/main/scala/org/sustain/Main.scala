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
import org.apache.spark.sql.RowFactory
import org.apache.spark.sql.functions.col

import java.util
import java.util.List

object Main {

  /* Entrypoint for the application */
  def main(args: Array[String]): Unit = {

    /* Global Variables */
    val SPARK_MASTER: String = "spark://lattice-150:8079"
    val APP_NAME: String = "Transfer Learning"
    val MONGO_URI: String = "mongodb://lattice-100:27018/"
    val MONGO_DB: String = "sustaindb"
    val MONGO_COLLECTION: String = "county_stats"
    val K: Int = 5
    val FEATURES: Array[String] = Array("median_household_income")

    /* Minimum Imports */
    import com.mongodb.spark._
    import com.mongodb.spark.MongoSpark
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.linalg.Vector
    import org.apache.spark.sql.{Dataset, Row, SparkSession}

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

    import sparkConnector.implicits._

    // Read collection into a DataSet, dropping null rows
    var collection: Dataset[Row] = MongoSpark.load(sparkConnector)
    collection = collection.select("GISJOIN", "median_household_income").na.drop()

    // Assemble features into single column
    var assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(FEATURES)
      .setOutputCol("features")
    var withFeaturesAssembled: Dataset[Row] = assembler.transform(collection)

    // Normalize features
    var minMaxScaler: MinMaxScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("normalized_features")
    var minMaxScalerModel: MinMaxScalerModel = minMaxScaler.fit(withFeaturesAssembled)
    var normalizedFeatures: Dataset[Row] = minMaxScalerModel.transform(withFeaturesAssembled)
    normalizedFeatures = normalizedFeatures.drop("features")
    normalizedFeatures = normalizedFeatures.withColumnRenamed("normalized_features", "features")
      .select("GISJOIN", "features")

    println(">>> With normalized features:\n")
    normalizedFeatures.show(10)

    // KMeans clustering
    val kMeans: KMeans = new KMeans().setK(K).setSeed(1L)
    val kMeansModel: KMeansModel = kMeans.fit(normalizedFeatures)
    val centers: Array[Vector] = kMeansModel.clusterCenters
    println(">>> Cluster centers:\n")
    centers.foreach { println }

    // Get cluster predictions
    var predictions: Dataset[Row] = kMeansModel.transform(normalizedFeatures)
    println(">>> Predictions centers:\n")

    predictions.show(10)

    /*
    +--------+-----------------------+--------------------+----------+
    | GISJOIN|median_household_income|            features|prediction|
    +--------+-----------------------+--------------------+----------+
    |G2100890|                43808.0|[0.23429567913195...|         1|
    |G2101610|                39192.0|[0.18957566363107...|         1|
    |G1300530|                48684.0|[0.2815345863204805]|         2|
    |G2500190|                83546.0|[0.6192792094555318]|         4|
    |G2102250|                37445.0|[0.17265064909901...|         3|
    |G2101770|                38835.0|[0.18611703158302...|         1|
    |G3900210|                50214.0|[0.29635729509784...|         2|
    |G2100670|                48779.0|[0.28245495059097...|         2|
    |G3901690|                49241.0|[0.28693082735903...|         2|
    |G3900170|                56253.0|[0.35486339856616...|         2|
    +--------+-----------------------+--------------------+----------+
     */

    println(">>> With Center")
    predictions = predictions.withColumn("center", col("prediction"))

    val withCenters: Dataset[Row] = predictions.map(row => {
      val prediction = row.getInt(4)
      (row.get(1), row.get(2), row.get(3), row.get(4), centers(prediction))
    }).toDF("GISJOIN", "features", "prediction", "center")
      .withColumn("distance", col("features").minus(col("center")))


    /*
    val distances = predictions.map(row => {

          val features: util.List[Float] = row.getList[Float](3)
          val prediction = row.getInt(4)
          val centersVect = centers(prediction)

          var sum = 0.0
          for (i <- centers.indices) {
            sum = sum + (features.get(i) - centers(i))
          }

      (row.get(1), row.get(2), row.get(3), col("features").minus(col("center")))
    })
    */

    withCenters.show(10)
  }

}
