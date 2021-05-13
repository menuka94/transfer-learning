package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.col
import com.mongodb.spark.config._

import java.io.{BufferedWriter, File, FileWriter}

class CentroidModel(sparkMasterC: String, mongoUriC: String, databaseC: String, collectionC: String,
                    labelC: String, featuresC: Array[String], gisJoinC: String, clusterIdC: Int,
                    sparkSessionC: SparkSession, profilerC: Profiler, iterationsOutput: String)
                    extends Thread with Serializable with Ordered[CentroidModel] {

  var linearRegression: LinearRegression = new LinearRegression()
  val sparkMaster: String = sparkMasterC
  val mongoUri: String = mongoUriC
  val database: String = databaseC
  val collection: String = collectionC
  val label: String = labelC
  val features: Array[String] = featuresC
  val gisJoin: String = gisJoinC
  val clusterId: Int = clusterIdC
  val sparkSession: SparkSession = sparkSessionC
  val profiler: Profiler = profilerC
  val iterationsOutputFile: String = iterationsOutput

  /**
   * Launched by the thread.start()
   */
  override def run(): Unit = {
    val trainTaskName: String = "CentroidModel;run(mongoUri=%s);gisJoin=%s;clusterId=%d".format(this.mongoUri, this.gisJoin, this.clusterId)
    val trainTaskId: Int = this.profiler.addTask(trainTaskName)
    println("\n\n" + trainTaskName)

    val readConfig: ReadConfig = ReadConfig(
      Map(
        "uri" -> this.mongoUri,
        "database" -> this.database,
        "collection" -> this.collection
      ), Some(ReadConfig(sparkSession))
    )

    val mongoCollection: Dataset[Row] = MongoSpark.load(this.sparkSession, readConfig).select(
      "gis_join", "year_month_day_hour", "timestep", "temp_surface_level_kelvin"
    )

    val transferLR: TransferLR = new TransferLR()
    this.linearRegression = transferLR.train(mongoCollection, this.label, this.features, this.iterationsOutputFile, this.gisJoin, this.clusterId,
    "CentroidModel", null, this.profiler)

    this.profiler.finishTask(trainTaskId, System.currentTimeMillis())
  }



  /**
   * Allows ordering of CentroidModel objects, sorted by ascending cluster id which the GISJoin belongs to.
   * @param that The other CentroidModel instance we are comparing ourselves to
   * @return 0 if the cluster ids are equal, 1 if our cluster id is greater than the other CentroidModel instance, and we
   *         should come after "that", and -1 if our cluster id is less than the other CentroidModel instance, and we
   *         should come before "that".
   */
  override def compare(that: CentroidModel): Int = {
    if (this.clusterId == that.clusterId)
      0
    else if (this.clusterId > that.clusterId)
      1
    else
      -1
  }

  /**
   * Overrides the toString method, for debugging model queues
   * @return String representation of Regression
   */
  override def toString: String = {
    "{%s|%d}".format(gisJoin, clusterId)
  }

}
