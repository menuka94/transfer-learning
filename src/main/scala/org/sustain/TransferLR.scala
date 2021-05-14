package org.sustain

import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

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
    //gisJoinCollection = gisJoinCollection.cache() // Cache Dataframe for just this GISJoin
    var Array(train, test): Array[Dataset[Row]] = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)
    profiler.finishTask(filterAndSplitTaskId, System.currentTimeMillis())

    // Create a linear regression model object and fit it to the training set
    // Copy the hyper-parameters from the already-trained centroid model for this cluster, if applicable
    val fitTaskName: String = "%s;Fit Training Set;gisJoin=%s;clusterId=%d".format(callerClass, gisJoin, clusterId)
    val fitTaskId: Int = profiler.addTask(fitTaskName)
    var linearRegression: LinearRegression = new LinearRegression()
      .setFitIntercept(true)
      .setLoss("huber")
      .setSolver("auto")
      .setRegParam(0.3)
      .setTol(0.001)
      .setMaxIter(100)
      .setEpsilon(1.35)
      .setElasticNetParam(0.0)
      .setStandardization(true)

    if (callerClass == "ClusterLRModels") {
      linearRegression = centroidEstimator.copy(new ParamMap())
    }
    val lrModel: LinearRegressionModel = linearRegression.fit(train)
    profiler.finishTask(fitTaskId, System.currentTimeMillis())

    val evaluateTaskName: String = "%s;Evaluate RMSE;gisJoin=%s;clusterId=%d".format(callerClass, gisJoin, clusterId)
    val evaluateTaskId: Int = profiler.addTask(evaluateTaskName)
    val totalIterations: Int = lrModel.summary.totalIterations
    println("\n\n>>> TOTAL ITERATIONS FOR GISJOIN %s: %d\n", gisJoin, totalIterations)
    writeTotalIterations(gisJoin, totalIterations, iterationsFilename, callerClass == "CentroidModel")

    // Use the model on the testing set, and evaluate results
    val lrPredictions: DataFrame = lrModel.transform(test)
    val evaluator: RegressionEvaluator = new RegressionEvaluator().setMetricName("rmse")
    println("\n\n>>> Test set RMSE for " + gisJoin + ": " + evaluator.evaluate(lrPredictions))
    profiler.finishTask(evaluateTaskId, System.currentTimeMillis())

    linearRegression
  }

  /**
   * Function for testing single-model training
   */
  def testTrain(): Unit = {

    val conf: SparkConf = new SparkConf()
      .setMaster("spark://lattice-100:8079")
      .setAppName("Test Single Model Training")
      .set("spark.executor.cores", "8")
      .set("spark.executor.memory", "10G")
      .set("spark.mongodb.input.uri", "mongodb://lattice-100:27018/") // default mongos router
      .set("spark.mongodb.input.database", "sustaindb") // sustaindb
      .set("spark.mongodb.input.collection", "noaa_nam_sharded") // noaa_nam
      //.set("spark.mongodb.input.partitioner", "MongoShardedPartitioner")
      //.set("spark.mongodb.input.partitionerOptions.shardkey", "gis_join")

    val sparkSession: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()


    val mongoCollection: Dataset[Row] = MongoSpark.load(sparkSession).select(
      "gis_join", "relative_humidity_percent", "timestep", "temp_surface_level_kelvin"
    )

    val regressionFeatures: Array[String] = Array("relative_humidity_percent", "surface_pressure_surface_level_pascal", "year_month_day_hour")
    val regressionLabel: String = "temp_surface_level_kelvin"
    val gisJoin: String = "G3100310"

    var gisJoinCollection: Dataset[Row] = mongoCollection.na.drop()
      .filter(
        col("gis_join") === gisJoin && col("timestep") === 0
      )
      .withColumnRenamed(regressionLabel, "label")

    // Write dataframe to csv for plotting raw data
//    gisJoinCollection.write.option("header", true)
//      .option("delimiter", ",")
//      .csv("/s/parsons/b/others/sustain/gis_join_collection.csv")

    // Assemble features column
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(regressionFeatures)
      .setOutputCol("features")
    gisJoinCollection = assembler.transform(gisJoinCollection)

    /*
    // Normalize features column
    val minMaxScaler: MinMaxScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("normalized_features")
    val minMaxScalerModel: MinMaxScalerModel = minMaxScaler.fit(gisJoinCollection)
    gisJoinCollection = minMaxScalerModel.transform(gisJoinCollection).drop("features").withColumnRenamed("normalized_features", "features")
    */

    // Persist the dataset
    gisJoinCollection = gisJoinCollection.persist()

    // Split into train/test sets
    val Array(train, test): Array[Dataset[Row]] = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)
    train.show()
    test.show()

    println("\n\nNUMBER OF ROWS: %d\n".format(gisJoinCollection.count()))
    gisJoinCollection.show()


    val tolerances: Array[Double] = Array(0.001)
    val regParams: Array[Double] = Array(0.3)
    val epsilons: Array[Double] = Array(1.1, 1.35, 1.5, 1.8)

    val bw = new BufferedWriter(new FileWriter(new File("lr_tests.csv")))
    bw.write("gis_join,total_iterations,tolerance,reg_param,epsilon,loss,test_rmse\n")

    for (tolerance <- tolerances) {

      for (regParam <- regParams) {

        for (epsilon <- epsilons) {

          val solver: String = "huber"
          val linearRegression: LinearRegression = new LinearRegression()
            .setFitIntercept(true)
            .setLoss("squaredError")
            .setSolver("auto")
            .setRegParam(regParam)
            .setTol(tolerance)
            .setMaxIter(100)
            .setEpsilon(epsilon)
            .setElasticNetParam(0.0)
            .setStandardization(true)

          val lrModel: LinearRegressionModel = linearRegression.fit(train)

          val totalIterations: Int = lrModel.summary.totalIterations

          val lrPredictions: Dataset[Row] = lrModel.transform(test)
          val evaluator: RegressionEvaluator = new RegressionEvaluator().setMetricName("rmse")
          val rmse: Double = evaluator.evaluate(lrPredictions)

          lrPredictions.show()

          println("\n\n>>> TOTAL ITERATIONS FOR GISJOIN %s: %d".format(gisJoin, totalIterations))
          println(">>> OBJECTIVE HISTORY:\n")
          lrModel.summary.objectiveHistory.foreach{ println }
          println(">>> TEST SET RMSE FOR TOL %f: %.4f".format(tolerance, rmse))
          println(">>> LR MODEL COEFFICIENTS: %s".format(lrModel.coefficients))
          println(">>> LR MODEL INTERCEPT: %.4f\n".format(lrModel.intercept))

          bw.write("%s,%d,%.4f,%.2f,%.2f,%s,%.4f\n".format(gisJoin,totalIterations,tolerance,regParam,epsilon,solver,rmse))
        } // End epsilons

      } // End regParams

    } // End tolerances
    bw.close()
    sparkSession.close()
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
