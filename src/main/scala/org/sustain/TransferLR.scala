package org.sustain

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import java.io.{BufferedWriter, File, FileWriter}

class TransferLR {

  def train(mongoCollection: Dataset[Row], label: String, features: Array[String], iterationsFilename: String,
            gisJoin: String, clusterId: Int, callerClass: String, centroidEstimator: LinearRegression,
            profiler: Profiler): LinearRegression = {

    // Filter the data down to just entries for a single GISJoin
    val filterAndSplitTaskName: String = "%s;Filter by GISJoin + Vector Transform + Split + Checkpoint;gisJoin=%s;clusterId=%d".format(callerClass, gisJoin, clusterId)
    val filterAndSplitTaskId: Int = profiler.addTask(filterAndSplitTaskName)

    var gisJoinCollection: Dataset[Row] = mongoCollection.na.drop()
      .filter(
        col("gis_join") === gisJoin && col("timestep") === 0
      )
      .withColumnRenamed(label, "label")

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")
    gisJoinCollection = assembler.transform(gisJoinCollection)

    // Split input into testing set and training set:
    // 80% training, 20% testing, with random seed of 42
    var Array(train, test): Array[Dataset[Row]] = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)
    train = train.localCheckpoint(true)
    profiler.finishTask(filterAndSplitTaskId, System.currentTimeMillis())

    // Create a linear regression model object and fit it to the training set
    // Copy the hyper-parameters from the already-trained centroid model for this cluster, if applicable
    val fitTaskName: String = "%s;Fit Training Set;gisJoin=%s;clusterId=%d".format(callerClass, gisJoin, clusterId)
    val fitTaskId: Int = profiler.addTask(fitTaskName)
    var linearRegression: LinearRegression = new LinearRegression()
    if (callerClass == "ClusterLRModels") {
      linearRegression = centroidEstimator.copy(new ParamMap())
    }
    val lrModel: LinearRegressionModel = linearRegression.fit(train)
    profiler.finishTask(fitTaskId, System.currentTimeMillis())

    val evaluateTaskName: String = "%s;Evaluate RMSE;gisJoin=%s;clusterId=%d".format(callerClass, gisJoin, clusterId)
    val evaluateTaskId: Int = profiler.addTask(evaluateTaskName)
    val totalIterations: Int = lrModel.summary.totalIterations
    writeTotalIterations(gisJoin, totalIterations, iterationsFilename, callerClass == "CentroidModel")

    // Use the model on the testing set, and evaluate results
    val lrPredictions: DataFrame = lrModel.transform(test)
    val evaluator: RegressionEvaluator = new RegressionEvaluator().setMetricName("rmse")
    println("\n\n>>> Test set RMSE for " + gisJoin + ": " + evaluator.evaluate(lrPredictions))
    profiler.finishTask(evaluateTaskId, System.currentTimeMillis())

    linearRegression
  }

  /**
   * Writes the total iterations until convergence of a model to file
   */
  def writeTotalIterations(gisJoin: String, iterations: Int, filename: String, isCentroid: Boolean): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File(filename),
        true
      )
    )
    bw.write("%s,%d,%s\n".format(gisJoin, iterations, isCentroid.toString))
    bw.close()
  }

}
