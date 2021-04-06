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

import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel}

class Main {

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
    import com.mongodb.spark.config.ReadConfig
    import com.mongodb.spark.sql.DefaultSource
    import com.mongodb.spark.MongoSpark
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.sql.DataFrame
    import org.apache.spark.sql.Dataset
    import org.apache.spark.sql.Row
    import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
    import org.apache.spark.ml.regression.LinearRegression
    import org.apache.spark.ml.regression.LinearRegressionModel
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    import org.apache.spark.sql.functions.col
    import org.apache.spark.ml.linalg.Vector

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
      .getOrCreate()

    import sparkConnector.implicits._ // For the $()-referenced columns

    // Read collection into a DataSet, dropping null rows
    var collection: Dataset[Row] = MongoSpark.load(sparkConnector)
    collection = collection.select("GISJOIN", "median_household_income").na.drop()

    println(">>> GOT HERE")
    collection.show(10)

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
  }

}
