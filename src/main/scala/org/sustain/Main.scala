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

import org.apache.spark.sql.RowFactory
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, DataTypes, FloatType}

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

    /* Read collection into a DataSet[Row], dropping null rows
      +--------+-----------------------+
      | GISJOIN|median_household_income|
      +--------+-----------------------+
      |G2100890|                43808.0|
      |G2101610|                39192.0|
      |G1300530|                48684.0|
      |G2500190|                83546.0|
      |G2102250|                37445.0|
      |G2101770|                38835.0|
      |G3900210|                50214.0|
      |G2100670|                48779.0|
      |G3901690|                49241.0|
      |G3900170|                56253.0|
      +--------+-----------------------+
     */
    var collection: Dataset[Row] = MongoSpark.load(sparkConnector)
    collection = collection.select("GISJOIN", "median_household_income").na.drop()
    collection.show(10)

    /* Assemble features into single column
      +--------+-----------------------+---------+
      | GISJOIN|median_household_income| features|
      +--------+-----------------------+---------+
      |G2100890|                43808.0|[43808.0]|
      |G2101610|                39192.0|[39192.0]|
      |G1300530|                48684.0|[48684.0]|
      |G2500190|                83546.0|[83546.0]|
      |G2102250|                37445.0|[37445.0]|
      |G2101770|                38835.0|[38835.0]|
      |G3900210|                50214.0|[50214.0]|
      |G2100670|                48779.0|[48779.0]|
      |G3901690|                49241.0|[49241.0]|
      |G3900170|                56253.0|[56253.0]|
      +--------+-----------------------+---------+
     */
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(FEATURES)
      .setOutputCol("features")
    val withFeaturesAssembled: Dataset[Row] = assembler.transform(collection)
    withFeaturesAssembled.show(10)

    /* Normalize features
      +--------+--------------------+
      | GISJOIN|            features|
      +--------+--------------------+
      |G2100890|[0.23429567913195...|
      |G2101610|[0.18957566363107...|
      |G1300530|[0.2815345863204805]|
      |G2500190|[0.6192792094555318]|
      |G2102250|[0.17265064909901...|
      |G2101770|[0.18611703158302...|
      |G3900210|[0.29635729509784...|
      |G2100670|[0.28245495059097...|
      |G3901690|[0.28693082735903...|
      |G3900170|[0.35486339856616...|
      +--------+--------------------+
     */
    val minMaxScaler: MinMaxScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("normalized_features")
    val minMaxScalerModel: MinMaxScalerModel = minMaxScaler.fit(withFeaturesAssembled)
    var normalizedFeatures: Dataset[Row] = minMaxScalerModel.transform(withFeaturesAssembled)
    normalizedFeatures = normalizedFeatures.drop("features")
    normalizedFeatures = normalizedFeatures.withColumnRenamed("normalized_features", "features")
      .select("GISJOIN", "features")
    println(">>> With normalized features:\n")
    normalizedFeatures.show(10)

    /* KMeans clustering centers
      [0.4287200541151439]
      [0.21934450125060204]
      [0.30615526209887767]
      [0.1292813510233636]
      [0.6361498632909213]
     */
    val kMeans: KMeans = new KMeans().setK(K).setSeed(1L)
    val kMeansModel: KMeansModel = kMeans.fit(normalizedFeatures)
    val centers: Array[Vector] = kMeansModel.clusterCenters
    println(">>> Cluster centers:\n")
    centers.foreach { println }

    /* Get cluster predictions
      +--------+--------------------+----------+
      | GISJOIN|            features|prediction|
      +--------+--------------------+----------+
      |G2100890|[0.23429567913195...|         1|
      |G2101610|[0.18957566363107...|         1|
      |G1300530|[0.2815345863204805]|         2|
      |G2500190|[0.6192792094555318]|         4|
      |G2102250|[0.17265064909901...|         3|
      |G2101770|[0.18611703158302...|         1|
      |G3900210|[0.29635729509784...|         2|
      |G2100670|[0.28245495059097...|         2|
      |G3901690|[0.28693082735903...|         2|
      |G3900170|[0.35486339856616...|         2|
      +--------+--------------------+----------+
     */
    var predictions: Dataset[Row] = kMeansModel.transform(normalizedFeatures)
    println(">>> Predictions centers:\n")
    predictions.show(10)

    /* Add "center" column
      +--------+---+--------------------+
      |      _1| _2|                  _3|
      +--------+---+--------------------+
      |G2100890|  1| 0.03780083758760514|
      |G2101610|  1|0.057190039499994794|
      |G1300530|  2|0.003867606680998721|
      |G2500190|  4|  0.2400979012681112|
      |G2102250|  3|0.017823481692243952|
      |G2101770|  1|0.058856226541719056|
      |G3900210|  2|0.005930970416158598|
      |G2100670|  2|0.003982928815943246|
      |G3901690|  2|0.004567911476835983|
      |G3900170|  2| 0.01836537152962727|
      +--------+---+--------------------+

     */
    println(">>> With Center")
    val arrayCol: ArrayType = DataTypes.createArrayType(FloatType)

    var withCenters = predictions.map( row => {
      val prediction:   Int    = row.getInt(2)        // Cluster prediction
      val featuresVect: Vector = row.getAs[Vector](1) // Normalized features
      val centersVect:  Vector = centers(prediction-1)  // Normalized cluster centers
      val distance = Vectors.sqdist(featuresVect, centersVect) // Squared dist between features and cluster centers

      (row.getString(0), row.getInt(2), distance) // (String, Int, Double)
    }).toDF("GISJOIN", "prediction", "distance")

    //withCenters = withCenters.withColumn("center", col("features"))
    withCenters.show(10)


//    val withCenters: Dataset[(String, util.List[Float], Int, Vector)] = predictions.map(row => {
//      val prediction = row.getInt(4)
//      (row.getString(1), row.getList[Float](2), row.getInt(3), centers(prediction))
//    })
      //.withColumn("distance", col("features").minus(col("center")))


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

  }

}
