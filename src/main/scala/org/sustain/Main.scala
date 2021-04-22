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
import org.apache.spark.sql.{DataFrame, RowFactory}
import org.apache.spark.sql.functions.{col, min, row_number}
import org.apache.spark.sql.types.{ArrayType, DataTypes, FloatType}

import java.util
import java.util.List

object Main {

  /* Entrypoint for the application */
  def main(args: Array[String]): Unit = {

    /* Global Variables */
    val SPARK_MASTER: String = "spark://lattice-100:8079"
    val APP_NAME: String = "Transfer Learning"
    val MONGO_URI: String = "mongodb://lattice-100:27018/"
    val MONGO_DB: String = "sustaindb"
    val MONGO_COLLECTION: String = "noaa_nam"
    val K: Int = 5
    val FEATURES: Array[String] = Array("temp_surface_level_kelvin")
    val CLUSTERING_YEAR_MONTH_DAY_HOUR: Long = 2010010100
    val CLUSTERING_TIMESTEP: Long = 0

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
      +--------+-------------------+--------+-------------------------+
      |gis_join|year_month_day_hour|timestep|temp_surface_level_kelvin|
      +--------+-------------------+--------+-------------------------+
      |G4804230|         2010010100|       0|        281.4640808105469|
      |G5600390|         2010010100|       0|        265.2140808105469|
      |G1701150|         2010010100|       0|        265.7140808105469|
      |G0601030|         2010010100|       0|        282.9640808105469|
      |G3701230|         2010010100|       0|        279.2140808105469|
      |G3700690|         2010010100|       0|        280.8390808105469|
      |G3701070|         2010010100|       0|        280.9640808105469|
      |G4803630|         2010010100|       0|        275.7140808105469|
      |G5108200|         2010010100|       0|        273.4640808105469|
      |G4801170|         2010010100|       0|        269.3390808105469|
      +--------+-------------------+--------+-------------------------+
     */
    var collection: Dataset[Row] = MongoSpark.load(sparkConnector)
    collection = collection.select("gis_join", "year_month_day_hour", "timestep", "temp_surface_level_kelvin")
      .na.drop()

    /* Take only 1 entry per GISJOIN across all timesteps for clustering
      +--------+-------------------------+
      |gis_join|temp_surface_level_kelvin|
      +--------+-------------------------+
      |G4804230|        281.4640808105469|
      |G5600390|        265.2140808105469|
      |G1701150|        265.7140808105469|
      |G0601030|        282.9640808105469|
      |G3701230|        279.2140808105469|
      |G3700690|        280.8390808105469|
      |G3701070|        280.9640808105469|
      |G4803630|        275.7140808105469|
      |G5108200|        273.4640808105469|
      |G4801170|        269.3390808105469|
      +--------+-------------------------+
     */
    val clusteringCollection: Dataset[Row] = collection.filter(
      col("year_month_day_hour") === CLUSTERING_YEAR_MONTH_DAY_HOUR && col("timestep") === CLUSTERING_TIMESTEP
    ).select("gis_join", "temp_surface_level_kelvin")
    clusteringCollection.show(10)

    /* Assemble features into single column
      +--------+-------------------------+-------------------+
      |gis_join|temp_surface_level_kelvin|           features|
      +--------+-------------------------+-------------------+
      |G4804230|        281.4640808105469|[281.4640808105469]|
      |G5600390|        265.2140808105469|[265.2140808105469]|
      |G1701150|        265.7140808105469|[265.7140808105469]|
      |G0601030|        282.9640808105469|[282.9640808105469]|
      |G3701230|        279.2140808105469|[279.2140808105469]|
      |G3700690|        280.8390808105469|[280.8390808105469]|
      |G3701070|        280.9640808105469|[280.9640808105469]|
      |G4803630|        275.7140808105469|[275.7140808105469]|
      |G5108200|        273.4640808105469|[273.4640808105469]|
      |G4801170|        269.3390808105469|[269.3390808105469]|
      +--------+-------------------------+-------------------+
     */
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(FEATURES)
      .setOutputCol("features")
    val withFeaturesAssembled: Dataset[Row] = assembler.transform(clusteringCollection)
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
      .select("gis_join", "features")
    println(">>> With normalized features:\n")
    normalizedFeatures.show(10)


    /* KMeans clustering centers
      [0.4287200541151439]
      [0.21934450125060204]
      [0.30615526209887767]
      [0.1292813510233636]
      [0.6361498632909213]
     */
    /*
    val kMeans: KMeans = new KMeans().setK(K).setSeed(1L)
    val kMeansModel: KMeansModel = kMeans.fit(normalizedFeatures)
    val centers: Array[Vector] = kMeansModel.clusterCenters
    println(">>> Cluster centers:\n")
    centers.foreach { println }
     */

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
    /*
    val predictions: Dataset[Row] = kMeansModel.transform(normalizedFeatures)
    println(">>> Predictions centers:\n")
    predictions.show(10)
     */

    /* Calculate distances to cluster center
      +--------+----------+--------------------+
      | GISJOIN|prediction|            distance|
      +--------+----------+--------------------+
      |G2100890|         1| 0.03780083758760514|
      |G2101610|         1|0.057190039499994794|
      |G1300530|         2|0.003867606680998721|
      |G2500190|         4|  0.2400979012681112|
      |G2102250|         3|0.017823481692243952|
      |G2101770|         1|0.058856226541719056|
      |G3900210|         2|0.005930970416158598|
      |G2100670|         2|0.003982928815943246|
      |G3901690|         2|0.004567911476835983|
      |G3900170|         2| 0.01836537152962727|
      +--------+----------+--------------------+
     */
    /*
    val distances: Dataset[Row] = predictions.map( row => {
      val prediction:   Int    = row.getInt(2)        // Cluster prediction
      val featuresVect: Vector = row.getAs[Vector](1) // Normalized features
      val centersVect:  Vector = centers(prediction)  // Normalized cluster centers
      val distance = Vectors.sqdist(featuresVect, centersVect) // Squared dist between features and cluster centers

      (row.getString(0), row.getInt(2), distance) // (String, Int, Double)
    }).toDF("GISJOIN", "prediction", "distance").as("distances")
    distances.show(100)
    distances.columns.foreach{ println }
     */


    /* Partition by prediction, find the minimum distance value, and pair back with original dataframe.
      +--------+----------+--------------------+
      | GISJOIN|prediction|            distance|
      +--------+----------+--------------------+
      |G2101010|         1|1.491467328893098...|
      |G5400570|         3|1.663968354711193...|
      |G3400370|         4|4.526691617257419E-6|
      |G3800230|         2|2.824090945215774E-8|
      |G4000170|         0|5.302094676808709E-9|
      +--------+----------+--------------------+
     */
    //val closestPoints = distances.groupBy("prediction").min("distance")
    //closestPoints.show(10)
    /*
    val closestPoints = Window.partitionBy("prediction").orderBy(col("distance").asc)
    distances.withColumn("row",row_number.over(closestPoints))
      .where($"row" === 1).drop("row")
      .show()


     */
  }

}
