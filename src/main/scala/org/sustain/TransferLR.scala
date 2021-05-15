package org.sustain

import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
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

    val regressionFeatures: Array[String] = Array("relative_humidity_percent")
    val regressionLabel: String = "temp_surface_level_kelvin"
    val gisJoin: String = "G3100310"

    var gisJoinCollection: Dataset[Row] = mongoCollection.na.drop()
      .filter(
        col("gis_join") === gisJoin && col("timestep") === 0
      )
      .withColumnRenamed(regressionLabel, "label")

    // Assemble features column
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(regressionFeatures)
      .setOutputCol("features")
    gisJoinCollection = assembler.transform(gisJoinCollection)

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
            .setSolver("l-bfgs")
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

  def testTrainTwo(): Unit = {

    val conf: SparkConf = new SparkConf()
      .setMaster("spark://lattice-100:8079")
      .setAppName("Test Single Model Training")
      .set("spark.executor.cores", "8")
      .set("spark.executor.memory", "10G")
      .set("spark.mongodb.input.uri", "mongodb://lattice-100:27018/") // default mongos router
      .set("spark.mongodb.input.database", "sustaindb") // sustaindb
      .set("spark.mongodb.input.collection", "noaa_nam_sharded") // noaa_nam

    val sparkSession: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()

    val regressionFeatures: Array[String] = Array("relative_humidity_percent")
    val regressionLabel: String = "temp_surface_level_kelvin"
    val gisJoin: String = "G3100310"

    // Load in Dataset; reduce it down to rows for this GISJoin at timestep 0; persist it for multiple operations
    var mongoCollection: Dataset[Row] = MongoSpark.load(sparkSession)
      .select("gis_join", "relative_humidity_percent", "timestep", "temp_surface_level_kelvin")
      .withColumnRenamed(regressionLabel, "label")
      .filter(col("gis_join") === gisJoin && col("timestep") === 0)
      .persist()

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(regressionFeatures)
      .setOutputCol("features")
    mongoCollection = assembler.transform(mongoCollection)

    // Split Dataset into train/test sets
    val Array(train, test): Array[Dataset[Row]] = mongoCollection.randomSplit(Array(0.8, 0.2), 42)

    // Create basic Linear Regression Estimator
    val linearRegression: LinearRegression = new LinearRegression()
      .setFitIntercept(true)
      .setMaxIter(10)
      .setLoss("squaredError")
      .setSolver("l-bfgs")
      .setStandardization(true)

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // With 3 values for tolerance, 3 values for regularization param, and 3 values for epsilon,
    // this grid will have 3 x 3 x 3 = 27 parameter settings for CrossValidator to choose from.
    val paramGrid: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(linearRegression.tol, Array(0.001, 0.01, 0.1))
      .addGrid(linearRegression.regParam, Array(0.0, 0.3, 0.1, 0.5))
      .addGrid(linearRegression.epsilon, Array(1.35, 1.1, 1.5))
      .build()

    // Establish a Regression Evaluator for RMSE
    val evaluator: RegressionEvaluator = new RegressionEvaluator()
      .setMetricName("rmse")

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val crossValidator: CrossValidator = new CrossValidator()
      .setEstimator(linearRegression)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)     // Use 3+ in practice
      .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

    // Run cross-validation, and choose the best set of parameters.
    val crossValidatorModel: CrossValidatorModel = crossValidator.fit(train)
    val bestLRModel: LinearRegressionModel = crossValidatorModel.bestModel.asInstanceOf[LinearRegressionModel]
    val bestLRRegParam: Double = bestLRModel.getRegParam
    val bestLRTol: Double = bestLRModel.getTol
    val bestLREpsilon: Double = bestLRModel.getEpsilon
    val bestLRIterations: Int = bestLRModel.summary.totalIterations

    println("\n\n>>> Best Params: tol=%.5f, regParam=%.2f, epsilon=%.2f\n".format(
      bestLRTol, bestLRRegParam, bestLREpsilon))

    println("\n\n>>> Summary History: totalIterations=%d, objectiveHistory:".format(bestLRIterations))
    bestLRModel.summary.objectiveHistory.foreach{println}

    val bestEstimator: LinearRegression = bestLRModel.parent.asInstanceOf[LinearRegression]

    // Fit a Linear Regression Estimator on all the data for the Centroid GISJoin,
    // we will transfer this trained estimator to other estimators for the cluster.
    val centroidLR: LinearRegression = new LinearRegression()
      .setFitIntercept(true)
      .setMaxIter(10)
      .setLoss("squaredError")
      .setSolver("l-bfgs")
      .setStandardization(true)
      .setRegParam(bestLRRegParam)
      .setTol(bestLRTol)
      .setEpsilon(bestLREpsilon)

    val linearRegressionModel: LinearRegressionModel = centroidLR.fit(train)

    // Record statistics on this model
    println("\n\n>>> Total Iterations: %d, Objective History:".format(linearRegressionModel.summary.totalIterations))
    linearRegressionModel.summary.objectiveHistory.foreach{println}


    // Make predictions on the testing Dataset, evaluate performance
    val predictions: Dataset[Row] = linearRegressionModel.transform(test)
    println("\n\n>>> Test set RMSE for " + gisJoin + ": " + evaluator.evaluate(predictions))


    // Unpersist Dataset to free up space
    mongoCollection.unpersist()
    //this.profiler.finishTask(trainTaskId, System.currentTimeMillis())
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
