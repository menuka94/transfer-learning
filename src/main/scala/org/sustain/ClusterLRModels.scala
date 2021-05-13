package org.sustain

import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Row, SparkSession}

class ClusterLRModels(sparkMasterC: String, mongoUriC: String, databaseC: String, collectionC: String, clusterIdC: Int,
                      gisJoinsC: Array[String], centroidEstimatorC: LinearRegression, centroidGisJoinC: String,
                      featuresC: Array[String], labelC: String, profilerC: Profiler, sparkSessionC: SparkSession, iterationsOutput: String)
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
  val iterationsOutputFile: String = iterationsOutput

  /**
   * Launched by the thread.start()
   */
  override def run(): Unit = {
    val trainTaskName: String = "ClusterLRModels;run(mongoUri=%s);gisJoin=%s;clusterId=%d".format(this.mongoUri, this.centroidGisJoin, this.clusterId)
    val trainTaskId: Int = this.profiler.addTask(trainTaskName)

    val readConfig: ReadConfig = ReadConfig(
      Map(
        "uri" -> this.mongoUri,
        "database" -> this.database,
        "collection" -> this.collection
      ), Some(ReadConfig(this.sparkSession))
    )

    val persistTaskName: String = ("ClusterLRModels;Cluster persist after select + drop null + filter + column rename;" +
      "gisJoin=%s;clusterId=%d").format(this.centroidGisJoin, this.clusterId)
    val persistTaskId: Int = this.profiler.addTask(persistTaskName)

    /* Read collection into a DataSet[Row], dropping null rows, filter by any GISJoins in the cluster, timestep 0, and
       rename the label column to "label"
      +--------+-------------------+--------+------------------+
      |gis_join|year_month_day_hour|timestep|             label|
      +--------+-------------------+--------+------------------+
      |G3600770|         2010011000|       0|258.02488708496094|
      |G3600770|         2010011000|       0|257.64988708496094|
      |G3600770|         2010011000|       0|257.39988708496094|
      |G3600770|         2010011000|       0|257.14988708496094|
      |G3600770|         2010011000|       0|257.39988708496094|
      |G3600770|         2010011000|       0|256.89988708496094|
      |G3600770|         2010011000|       0|256.64988708496094|
      |G3600770|         2010011000|       0|256.77488708496094|
      |G3600770|         2010011000|       0|257.14988708496094|
      |G3600770|         2010011000|       0|256.77488708496094|
      +--------+-------------------+--------+------------------+
     */
    var mongoCollection: Dataset[Row] = MongoSpark.load(this.sparkSession, readConfig).select(
      "gis_join", "year_month_day_hour", "timestep", "temp_surface_level_kelvin"
    ).na.drop().filter(
      col("gis_join").isInCollection(this.gisJoins) && col("timestep") === 0
    ).withColumnRenamed(this.label, "label").persist()
    this.profiler.finishTask(persistTaskId, System.currentTimeMillis())

    // Iterate over all gisJoins in this collection, build models for each from persisted collection
    val transferLR: TransferLR = new TransferLR()
    this.gisJoins.foreach(
      gisJoin => {
        transferLR.train(mongoCollection, this.label, this.features, this.iterationsOutputFile, gisJoin, this.clusterId,
        "ClusterLRModels", this.centroidEstimator, this.profiler)
      }
    )

    this.profiler.finishTask(trainTaskId, System.currentTimeMillis())
    mongoCollection.unpersist()
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
