package org.sustain

import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset, Row, RowFactory}
import org.apache.spark.sql.functions.{col, collect_list, min, row_number, struct}
import org.apache.spark.ml.regression.LinearRegression
import com.mongodb.spark.MongoSpark
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorDisassembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Dataset, Row, SparkSession}

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

@SerialVersionUID(114L)
class Experiment() extends Serializable {

  def transferLearning(sparkMaster: String, appName: String, mongosRouters: Array[String], mongoPort: String,
                       database: String, collection: String, clusteringFeatures: Array[String], clusteringYMDH: Long,
                       clusteringTimestep: Long, clusteringK: Int, regressionFeatures: Array[String],
                       regressionLabel: String, pcaFeatures: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName(appName)
      .set("spark.executor.cores", "8")
      .set("spark.executor.memory", "20G")
      .set("spark.mongodb.input.uri", "mongodb://%s:%s/".format(mongosRouters(0), mongoPort))
      .set("spark.mongodb.input.database", database)
      .set("spark.mongodb.input.collection", collection)
      .set("mongodb.keep_alive_ms", "100000") // Important! Default is 5000ms, and stream will prematurely close

    // Create the SparkSession and ReadConfig
    val sparkConnector: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate() // For the $()-referenced columns

    import sparkConnector.implicits._

    val pca: PrincipleComponentAnalysis = new PrincipleComponentAnalysis()

    /* Run PCA on the collection and get the DF containing the principle components
      +--------+--------------------+--------------------+
      |gis_join|            features|         pcaFeatures|
      +--------+--------------------+--------------------+
      |G1200870|[0.54405509418675...|[0.80492832393268...|
      |G1200870|[0.55296738910269...|[0.80857487657638...|
      |G1200870|[0.55276483694551...|[0.76926659597088...|
      |G1200870|[0.55377759773141...|[0.79674422064382...|
      |G1200870|[0.54628316791573...|[0.71710709966826...|
      |G1200870|[0.55448653028154...|[0.77397137809210...|
      |G1200870|[0.55600567146040...|[0.68385808112508...|
      |G1200870|[0.55772736479643...|[0.66660908921373...|
      |G1200870|[0.55499291067449...|[0.75565165269005...|
      |G1200870|[0.55468908243872...|[0.72406202099241...|
      |G1200870|[0.55701843224630...|[0.70259295830020...|
      |G1200870|[0.55894267773951...|[0.65973792208690...|
      ...
      +--------+--------------------+--------------------+
     */
    val pcaDF: Dataset[Row] = pca.runPCA(sparkConnector, pcaFeatures)
    val disassembler = new VectorDisassembler().setInputCol("pcaFeatures")
    disassembler.transform(pcaDF).show(20)


/*
    val kMeansClustering: KMeansClustering = new KMeansClustering()
    val gisJoinCenters: Array[(String, Int)] = kMeansClustering.runClustering(sparkConnector, pcaDF, clusteringFeatures,
      clusteringYMDH, clusteringTimestep, clusteringK)


 */


    // Create LR models for cluster centroid GISJoins
    /*
    val centroidModels: Array[CentroidModel] = new Array[CentroidModel](gisJoinCenters.length)
    for (i <- gisJoinCenters.indices) {
      val gisJoin: String = gisJoinCenters(i)._1
      val clusterId: Int = gisJoinCenters(i)._2
      val mongoHost: String = mongosRouters(clusterId % mongosRouters.length) // choose a mongos router
      val centroidModel: CentroidModel = new CentroidModel(sparkMaster, mongoHost, mongoPort,
        database, collection, regressionLabel, regressionFeatures, gisJoin, clusterId)
      centroidModels(clusterId) = centroidModel
    }

    try {
      // Kick off training of LR models for center GISJoins
      centroidModels.foreach(model => model.start())

      // Wait until models are done being trained
      centroidModels.foreach(model => model.join())
    } catch {
      case e: java.lang.IllegalMonitorStateException => println("\n\nn>>>Caught IllegalMonitorStateException!")
    }

    println("\n\n>>> Initial center models done training\n")

    // Sort trained models by their predicted cluster ID
    scala.util.Sorting.quickSort(centroidModels)

     */

    /* Group together GISJoins which belong to the same cluster
      +----------+----------------------+
      |prediction|collect_list(gis_join)|
      +----------+----------------------+
      |        31|  [G3800670, G38009...|
      |        53|  [G5500510, G35000...|
      |        34|  [G3100250, G31013...|
      |        28|  [G1701150, G36000...|
      |        26|  [G1301790, G13022...|
      |        27|  [G4100390, G13000...|
      |        44|  [G5500150, G55008...|
      |        12|  [G7200540, G72001...|
      |        22|  [G3101030, G31014...|
      |        47|  [G4800890, G48046...|
      |         1|  [G1700990, G55012...|
      ...
      +----------+----------------------+
     */

    /*
    val clusterRows: Dataset[Row] = predictions.select("gis_join", "prediction")
      .groupBy(col("prediction"))
      .agg(collect_list("gis_join"))

    // Collect clustered GISJoins into memory
    val clusters: Array[(Int, Array[String])] = clusterRows.collect()
      .map(
        row => {
          val gisJoins: Array[String] = row.getList[String](1).asScala.toArray
          (row.getInt(0), gisJoins)
        }
      )

    // Build K queues of models to be trained, 1 queue per cluster
    // For 3192 GISJoins this is sqrt(3192) = ~56 queues, each with ~57 models to be trained (since cluster sizes vary,
    // some queues may be shorter and others larger)
    val clustersQueues: Array[ClusterLRModels] = new Array[ClusterLRModels](gisJoinCenters.length)
    clusters.foreach(
      cluster => { // (31, [G3800670, G38009, ...])
        val clusterId: Int = cluster._1
        val gisJoins: Array[String] = cluster._2
        val center: (String, Int) = gisJoinCenters(clusterId)
        val centerGisJoin: String = center._1
        val trainedRegression: CentroidModel = centroidModels(clusterId)
        val trainedModel: LinearRegression = trainedRegression.linearRegression
        val mongoRouterHost: String = mongosRouters(clusterId % mongosRouters.length)
        clustersQueues(clusterId) = new ClusterLRModels(sparkMaster, mongoRouterHost, mongoPort, database, collection,
          clusterId, gisJoins, trainedModel, centerGisJoin, regressionFeatures, regressionLabel, mongoCollection)
      }
    )

    // --- DEBUGGING ---
    println("\n\n>>> MODEL QUEUES <<<\n")
    clustersQueues.foreach{ println }


    try {
      clustersQueues.foreach(queue => queue.start())
      clustersQueues.foreach(queue => queue.wait())
    } catch {
      case e: java.lang.IllegalMonitorStateException => println("\n\nn>>>Caught IllegalMonitorStateException!")
    }

     */
  }

}
