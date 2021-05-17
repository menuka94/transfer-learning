package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import java.io.{BufferedWriter, File, FileWriter}

class TransferLR {

  def train(mongoCollection: Dataset[Row], centroidEstimator: LinearRegression, clusterStatsCSVFilename: String,
            gisJoin: String, clusterId: Int, profiler: Profiler): Unit = {

    // >>> Begin Task for single cluster model's train() function
    val trainTaskName: String = "ClusterLRModels;train();gisJoin=%s;clusterId=%d"
      .format(gisJoin, clusterId)
    val trainTaskId: Int = profiler.addTask(trainTaskName)

    // Filter the data down to just entries for a single GISJoin
    val gisJoinCollection: Dataset[Row] = mongoCollection.filter(
      col("gis_join") === gisJoin && col("timestep") === 0)

    val numRecords: Long = gisJoinCollection.count()

    // Split input into testing set and training set:
    // 80% training, 20% testing, with random seed of 42
    //gisJoinCollection = gisJoinCollection.cache() // Cache Dataframe for just this GISJoin
    val Array(train, test): Array[Dataset[Row]] = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)

    // Copy the hyper-parameters from the already-trained centroid model for this cluster to a new LR Estimator
    val linearRegression: LinearRegression = centroidEstimator.copy(new ParamMap())
    val begin: Long = System.currentTimeMillis()
    val lrModel: LinearRegressionModel = linearRegression.fit(train)
    val end: Long = System.currentTimeMillis()
    val iterations: Int = lrModel.summary.totalIterations

    // Use the model on the testing set, and evaluate results
    val predictions: DataFrame = lrModel.transform(test)
    val evaluator: RegressionEvaluator = new RegressionEvaluator().setMetricName("rmse")
    val rmse: Double = evaluator.evaluate(predictions)
    println("\n\n>>> Test set RMSE for %s: %f\n".format(gisJoin, rmse))

    writeClusterModelStats(clusterStatsCSVFilename, gisJoin, clusterId, end-begin, rmse, iterations, numRecords)

    // <<< End Task for single cluster model's train() function
    profiler.finishTask(trainTaskId, System.currentTimeMillis())
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
      .set("spark.executor.memory", "16G")
      .set("spark.mongodb.input.uri", "mongodb://lattice-100:27018/") // default mongos router
      .set("spark.mongodb.input.database", "sustaindb") // sustaindb
      .set("spark.mongodb.input.collection", "noaa_nam_sharded") // noaa_nam
      .set("spark.mongodb.input.readPreference", "secondary")

    val sparkSession: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()

    val regressionFeatures: Array[String] = Array(
      "relative_humidity_percent",
      "orography_surface_level_meters",
      "relative_humidity_percent",
      "10_metre_u_wind_component_meters_per_second",
      "pressure_pascal",
      "visibility_meters",
      "total_cloud_cover_percent",
      "10_metre_u_wind_component_meters_per_second",
      "10_metre_v_wind_component_meters_per_second"
    )
    val regressionLabel: String = "temp_surface_level_kelvin"
    val gisJoin: String = "G3100310"

    // Load in Dataset; reduce it down to rows for this GISJoin at timestep 0; persist it for multiple operations
    var mongoCollection: Dataset[Row] = MongoSpark.load(sparkSession)
      .select(
        "gis_join",
        "timestep",
        "temp_surface_level_kelvin",
        "relative_humidity_percent",
        "orography_surface_level_meters",
        "relative_humidity_percent",
        "10_metre_u_wind_component_meters_per_second",
        "pressure_pascal",
        "visibility_meters",
        "total_cloud_cover_percent",
        "10_metre_u_wind_component_meters_per_second",
        "10_metre_v_wind_component_meters_per_second")
      .withColumnRenamed(regressionLabel, "label")
      .filter(col("gis_join") === gisJoin && col("timestep") === 0)
      .persist()

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(regressionFeatures)
      .setOutputCol("features")
    mongoCollection = assembler.transform(mongoCollection).persist()

    // Split Dataset into train/test sets
    val Array(train, test): Array[Dataset[Row]] = mongoCollection.randomSplit(Array(0.8, 0.2), 42)

    // Create basic Linear Regression Estimator
    val linearRegression: LinearRegression = new LinearRegression()
      .setFitIntercept(true)
      .setMaxIter(10)
      .setLoss("squaredError")
      .setSolver("l-bfgs")
      .setStandardization(true)

    // Run cross-validation, and choose the best set of parameters.
    val lrModel: LinearRegressionModel = linearRegression.fit(train)
    val iterations: Int = lrModel.summary.totalIterations

    println("\n\n>>> Summary History: totalIterations=%d, objectiveHistory:".format(iterations))
    lrModel.summary.objectiveHistory.foreach{println}

    // Establish a Regression Evaluator for RMSE
    val evaluator: RegressionEvaluator = new RegressionEvaluator()
      .setMetricName("rmse")
    val predictions: Dataset[Row] = lrModel.transform(test)
    val rmse: Double = evaluator.evaluate(predictions)

    // Make predictions on the testing Dataset, evaluate performance
    println("\n\n>>> Test set RMSE: %f".format(rmse))
    mongoCollection.unpersist()
  }

  /**
   * Writes the modeling stats for a single model to a CSV file
   */
  def writeClusterModelStats(filename: String, gisJoin: String, clusterId: Int, time: Long, rmse: Double,
                             iterations: Int, numRecords: Long): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File(filename),
        true
      )
    )
    bw.write("%s,%d,%d,%d,%f,%d\n".format(gisJoin, clusterId, time, iterations, rmse, numRecords))
    bw.close()
  }

}
