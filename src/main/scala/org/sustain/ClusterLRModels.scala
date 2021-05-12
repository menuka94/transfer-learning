package org.sustain

import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import java.io.{BufferedWriter, File, FileWriter, PrintWriter}

class ClusterLRModels(sparkMasterC: String, mongoUriC: String, databaseC: String, collectionC: String, clusterIdC: Int,
                      gisJoinsC: Array[String], centroidEstimatorC: LinearRegression, centroidGisJoinC: String,
                      featuresC: Array[String], labelC: String, profilerC: Profiler, sparkSessionC: SparkSession)
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
    var mongoCollection: Dataset[Row] = MongoSpark.load(this.sparkSession, readConfig)
    mongoCollection = mongoCollection.select("gis_join", "year_month_day_hour", "timestep", "temp_surface_level_kelvin")
      .na.drop().filter(
      col("gis_join").isInCollection(this.gisJoins) && col("timestep") === 0
    ).withColumnRenamed(this.label, "label")
    mongoCollection.persist() // Persist collection for reuse
    this.profiler.finishTask(persistTaskId, System.currentTimeMillis())

    // Iterate over all gisJoins in this collection, build models for each from persisted collection
    this.gisJoins.foreach(
      gisJoin => {

        // Filter the data down to just entries for a single GISJoin
        val splitAndFitTaskName: String = "ClusterLRModels;Filter + split test/train + LR fit;gisJoin=%s;clusterId=%d".format(gisJoin, this.clusterId)
        val splitAndFitTaskId: Int = this.profiler.addTask(splitAndFitTaskName)

        var gisJoinCollection: Dataset[Row] = mongoCollection.filter(col("gis_join") === gisJoin)
          .withColumnRenamed(this.label, "label")

        val assembler: VectorAssembler = new VectorAssembler()
          .setInputCols(this.features)
          .setOutputCol("features")
        gisJoinCollection = assembler.transform(gisJoinCollection)

        // Split input into testing set and training set:
        // 80% training, 20% testing, with random seed of 42
        val Array(train, test): Array[Dataset[Row]] = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)

        // Copy the hyper-parameters from the already-trained centroid model for this cluster
        val linearRegression: LinearRegression = centroidEstimator.copy(new ParamMap())

        // Create a linear regression model object and fit it to the training set
        val lrModel: LinearRegressionModel = linearRegression.fit(train)
        this.profiler.finishTask(splitAndFitTaskId, System.currentTimeMillis())

        val totalIterations: Int = lrModel.summary.totalIterations
        writeTotalIterations(gisJoin, totalIterations)

        // Use the model on the testing set, and evaluate results
        val evaluateTaskName: String = "ClusterLRModels;Evaluate RMSE;gisJoin=%s;clusterId=%d".format(gisJoin, this.clusterId)
        val evaluateTaskId: Int = this.profiler.addTask(evaluateTaskName)
        val lrPredictions: DataFrame = lrModel.transform(test)
        val evaluator: RegressionEvaluator = new RegressionEvaluator().setMetricName("rmse")
        println("\n\n>>> Test set RMSE for " + gisJoin + ": " + evaluator.evaluate(lrPredictions))
        this.profiler.finishTask(evaluateTaskId, System.currentTimeMillis())
      }
    )

    this.profiler.finishTask(trainTaskId, System.currentTimeMillis())
    mongoCollection.unpersist()
  }

  /**
   * Writes the total iterations until convergence of a model to file
   */
  def writeTotalIterations(gisJoin: String, iterations: Int): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File("train_iterations.csv"),
        true
      )
    )
    bw.write("%s,%d,false\n".format(gisJoin, iterations))
    bw.close()
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
