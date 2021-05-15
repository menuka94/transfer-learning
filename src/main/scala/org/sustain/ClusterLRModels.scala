package org.sustain

import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Row, SparkSession}

class ClusterLRModels(sparkMasterC: String, mongoUriC: String, databaseC: String, collectionC: String, clusterIdC: Int,
                      gisJoinsC: Array[String], centroidEstimatorC: LinearRegression, centroidGisJoinC: String,
                      featuresC: Array[String], labelC: String, profilerC: Profiler, sparkSessionC: SparkSession,
                      clusterStatsCSVFilenameC: String)
                      extends Thread with Serializable {

  val sparkMaster: String = sparkMasterC
  val mongoUri: String = mongoUriC
  val database: String = databaseC
  val collection: String = collectionC
  val clusterId: Int = clusterIdC
  val centroidGisJoin: String = centroidGisJoinC
  val gisJoins: Array[String] = gisJoinsC
  val centroidEstimator: LinearRegression = centroidEstimatorC
  val features: Array[String] = featuresC
  val label: String = labelC
  val profiler: Profiler = profilerC
  val sparkSession: SparkSession = sparkSessionC
  val clusterStatsCSVFilename: String = clusterStatsCSVFilenameC

  /**
   * Launched by the thread.start()
   */
  override def run(): Unit = {
    // >>> Begin Task for cluster's run() function
    val runTaskName: String = "ClusterLRModels;run();gisJoin=%s;clusterId=%d".format(this.centroidGisJoin, this.clusterId)
    val runTaskId: Int = this.profiler.addTask(runTaskName)

    val readConfig: ReadConfig = ReadConfig(
      Map(
        "uri" -> this.mongoUri,
        "database" -> this.database,
        "collection" -> this.collection
      ), Some(ReadConfig(this.sparkSession))
    )

    // >>> Begin Task for cluster's Dataframe operations
    val persistTaskName: String = ("ClusterLRModels;Cluster persist and count after select + drop null + filter + column rename;" +
      "gisJoin=%s;clusterId=%d").format(this.centroidGisJoin, this.clusterId)
    val persistTaskId: Int = this.profiler.addTask(persistTaskName)

    var mongoCollection: Dataset[Row] = MongoSpark.load(this.sparkSession, readConfig).select(
      "gis_join", "relative_humidity_percent", "timestep", "temp_surface_level_kelvin"
    ).na.drop().filter(
      col("gis_join").isInCollection(this.gisJoins) && col("timestep") === 0
    ).withColumnRenamed(this.label, "label")

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(this.features)
      .setOutputCol("features")
    mongoCollection = assembler.transform(mongoCollection).persist()
    val numRecords: Long = mongoCollection.count()

    // <<< End Task for cluster's Dataframe Operations
    this.profiler.finishTask(persistTaskId, System.currentTimeMillis())

    // >>> Begin Task for training all models in cluster
    val trainAllClusterModelsTaskName: String = ("ClusterLRModels;Train all cluster models;" +
      "gisJoin=%s;clusterId=%d;numModels=%d,numRecords=%d").format(
      this.centroidGisJoin, this.clusterId, this.gisJoins.length, numRecords)
    val trainAllClusterModelsTaskId: Int = this.profiler.addTask(trainAllClusterModelsTaskName)

    // Iterate over all gisJoins in this collection, build models for each from persisted collection
    val transferLR: TransferLR = new TransferLR()
    this.gisJoins.foreach(
      gisJoin => {
        transferLR.train(mongoCollection, this.centroidEstimator, this.clusterStatsCSVFilename, gisJoin, this.clusterId,
          this.profiler)
      }
    )

    // <<< End Task for training all models in cluster
    this.profiler.finishTask(trainAllClusterModelsTaskId, System.currentTimeMillis())

    // Unpersist Dataset to free up space
    mongoCollection.unpersist()

    // <<< End Task for cluster's run() function
    this.profiler.finishTask(runTaskId, System.currentTimeMillis())
  }

  /**
   * Overrides the toString method, for debugging model queues
   * @return String representation of Regression
   */
  override def toString: String = {
    var retVal: String = "Cluster ID: [%d]: [ ".format(clusterId)
    gisJoins.foreach(gisJoin => retVal += gisJoin + " ")
    retVal += "]\n"
    retVal
  }

}
