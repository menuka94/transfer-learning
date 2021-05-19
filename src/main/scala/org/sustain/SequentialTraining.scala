package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.col

import java.io.{BufferedWriter, File, FileWriter}
import scala.collection.mutable.ArrayBuffer

class SequentialTraining() {

  def runNonTransferLearnedModels(sparkMaster: String, mongoUri: String, database: String, collection: String,
                                  gisJoins: Array[String], features: Array[String], label: String): Unit = {

    writeHeader("no_tl_model_stats.csv")

    val conf: SparkConf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("Sequential, Non-TL LR Models")
      .set("spark.executor.cores", "8")
      .set("spark.executor.memory", "16G")
      .set("spark.mongodb.input.uri", mongoUri) // default mongos router
      .set("spark.mongodb.input.database", database) // sustaindb
      .set("spark.mongodb.input.collection", collection) // noaa_nam
      .set("spark.mongodb.input.readPreference", "nearest")

    val sparkSession: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()

    val gisJoinBatches: Array[ArrayBuffer[String]] = loadDefaultClusters(gisJoins)

    for (i <- gisJoinBatches.indices) {
      println("\n\n>>> Working on batch %d\n".format(i))
      val batch: ArrayBuffer[String] = gisJoinBatches(i)

      // Load in Dataset and persist it
      var mongoCollection: Dataset[Row] = MongoSpark.load(sparkSession)
        .select(
          "gis_join",
          "timestep",
          "temp_surface_level_kelvin",
          "relative_humidity_percent",
          "orography_surface_level_meters",
          "relative_humidity_percent",
          "pressure_pascal",
          "visibility_meters",
          "total_cloud_cover_percent",
          "10_metre_u_wind_component_meters_per_second",
          "10_metre_v_wind_component_meters_per_second")
        .withColumnRenamed(label, "label")
        .filter(
          col("timestep") === 0 && col("gis_join").isInCollection(batch)
        )

      val assembler: VectorAssembler = new VectorAssembler()
        .setInputCols(features)
        .setOutputCol("features")

      mongoCollection = assembler.transform(mongoCollection)
        .select("gis_join", "features", "label")
        .persist()

      // Sequentially train all models, without transfer-learning
      batch.foreach(
        gisJoin => {

          // Filter down to just this GISJoin
          val gisJoinCollection = mongoCollection.filter(
            col("gis_join") === gisJoin
          )

          // Split Dataset into train/test sets
          val Array(train, test): Array[Dataset[Row]] = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)

          // Create basic Linear Regression Estimator
          val linearRegression: LinearRegression = new LinearRegression()
            .setFitIntercept(true)
            .setMaxIter(100)
            .setLoss("squaredError")
            .setSolver("l-bfgs")
            .setStandardization(true)

          val numRecords: Long = gisJoinCollection.count()

          // Fit on training set
          val begin: Long = System.currentTimeMillis()
          val lrModel: LinearRegressionModel = linearRegression.fit(train)
          val end: Long = System.currentTimeMillis()
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
          writeModelStats("no_tl_model_stats.csv", gisJoin, System.currentTimeMillis(), "false", end-begin, iterations, rmse, numRecords)
          mongoCollection.unpersist()
        }
      )
    }

  }

  def runTransferLearnedModels(sparkMaster: String, mongoUri: String, database: String, collection: String,
                               pcaClusters: Array[PCACluster], features: Array[String], label: String): Unit = {

    writeHeader("tl_model_stats.csv")

    var clusterCount: Int = 0
    var modelCount: Int = 0

    scala.util.Sorting.quickSort(pcaClusters) // Sort by cluster id

    val conf: SparkConf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("Sequential, Non-TL LR Models")
      .set("spark.executor.cores", "8")
      .set("spark.executor.memory", "16G")
      .set("spark.mongodb.input.uri", mongoUri) // default mongos router
      .set("spark.mongodb.input.database", database) // sustaindb
      .set("spark.mongodb.input.collection", collection) // noaa_nam
      .set("spark.mongodb.input.readPreference", "nearest")

    val sparkSession: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()

    for (pcaCluster: PCACluster <- pcaClusters) {

      // Gather ALL gisJoins in this cluster, including centroid
      val clusterGisJoins: Array[String] = loadPCACluster(pcaCluster)

      // Load in Dataset, VectorAssemble the features column, and persist it
      var mongoCollection: Dataset[Row] = MongoSpark.load(sparkSession)
        .select(
          "gis_join",
          "timestep",
          "temp_surface_level_kelvin",
          "relative_humidity_percent",
          "orography_surface_level_meters",
          "relative_humidity_percent",
          "pressure_pascal",
          "visibility_meters",
          "total_cloud_cover_percent",
          "10_metre_u_wind_component_meters_per_second",
          "10_metre_v_wind_component_meters_per_second")
        .withColumnRenamed(label, "label")
        .filter(
          col("timestep") === 0 && col("gis_join").isInCollection(clusterGisJoins)
        )

      val assembler: VectorAssembler = new VectorAssembler()
        .setInputCols(features)
        .setOutputCol("features")

      mongoCollection = assembler.transform(mongoCollection)
        .select("gis_join", "features", "label")
        .persist()

      // --------- BEGIN CLUSTER CENTROID TRAINING -----------

      println("\n\n>>> Training Cluster %d, Model %d\n".format(clusterCount, modelCount))

      var gisJoinCollection: Dataset[Row] = mongoCollection.filter(
        col("gis_join") === pcaCluster.centerGisJoin
      )

      // Split Dataset into train/test sets
      val Array(train, test): Array[Dataset[Row]] = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)

      // Create basic Linear Regression Estimator
      val linearRegression: LinearRegression = new LinearRegression()
        .setFitIntercept(true)
        .setMaxIter(100)
        .setLoss("squaredError")
        .setSolver("l-bfgs")
        .setStandardization(true)

      var numRecords: Long = gisJoinCollection.count()

      // We use a ParamGridBuilder to construct a grid of parameters to search over.
      // With 3 values for tolerance, 3 values for regularization param, and 3 values for epsilon,
      // this grid will have 3 x 4 x 3 = 36 parameter settings for CrossValidator to choose from.
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
      val begin: Long = System.currentTimeMillis()
      val crossValidatorModel: CrossValidatorModel = crossValidator.fit(train)
      val end: Long = System.currentTimeMillis()

      val bestLRModel: LinearRegressionModel = crossValidatorModel.bestModel.asInstanceOf[LinearRegressionModel]
      val bestLinearRegression: LinearRegression = bestLRModel.parent.asInstanceOf[LinearRegression]
      val bestLRRegParam: Double = bestLRModel.getRegParam
      val bestLRTol: Double = bestLRModel.getTol
      val bestLREpsilon: Double = bestLRModel.getEpsilon
      val bestLRIterations: Int = bestLRModel.summary.totalIterations

      // Make predictions on the testing Dataset, evaluate performance
      val predictions: Dataset[Row] = bestLRModel.transform(test)
      val rmse: Double = evaluator.evaluate(predictions)

      println("\n\n>>> Test set RMSE for %s: %f\n".format(pcaCluster.centerGisJoin, rmse))

      // Write stats for centroid training
      writeModelStats("tl_model_stats.csv", pcaCluster.centerGisJoin, System.currentTimeMillis(), "true", end-begin, bestLRIterations, rmse, numRecords)
      modelCount += 1
      // --------- END CLUSTER CENTROID TRAINING -----------

      // --------- BEGIN ALL CLUSTER MODELS TRAINING -----------

      for (gisJoin: String <- pcaCluster.clusterGisJoins) {

        println("\n\n>>> Training Cluster %d, Model %d\n".format(clusterCount, modelCount))

        // Filter down to just this GISJoin
        gisJoinCollection = mongoCollection.filter(
          col("gis_join") === gisJoin
        )

        // Split Dataset into train/test sets
        val Array(train, test) = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)

        // Copy best model estimator to new one for this model
        val tlLinearRegression: LinearRegression = bestLinearRegression.copy(new ParamMap())

        numRecords = gisJoinCollection.count()

        // Fit on training set
        val begin: Long = System.currentTimeMillis()
        val lrModel: LinearRegressionModel = linearRegression.fit(train)
        val end: Long = System.currentTimeMillis()
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
        writeModelStats("tl_model_stats.csv", gisJoin, System.currentTimeMillis(), "false", end-begin, iterations, rmse, numRecords)
        modelCount += 1
      }

      // --------- END ALL CLUSTER MODELS TRAINING -----------

      mongoCollection.unpersist()
      clusterCount += 1
    }

  }

  def loadPCACluster(pcaCluster: PCACluster): Array[String] = {

    // Allocate Array to fit all normal models + centroid
    val totalGisJoins: Array[String] = new Array[String](pcaCluster.clusterGisJoins.length + 1)
    totalGisJoins(0) = pcaCluster.centerGisJoin
    for (i <- pcaCluster.clusterGisJoins.indices) {
      totalGisJoins(i+1) = pcaCluster.clusterGisJoins(i)
    }

    totalGisJoins
  }

  def loadDefaultClusters(gisJoins: Array[String]): Array[ArrayBuffer[String]] = {
    val clusters: Array[ArrayBuffer[String]] = new Array[ArrayBuffer[String]](56)
    for (i <- clusters.indices) {
      clusters(i) = new ArrayBuffer[String]()
    }
    for (i <- gisJoins.indices) {
      val clusterIndex: Int = i % 56
      clusters(clusterIndex) += gisJoins(i)
    }

    for (i <- clusters.indices) {
      println("Cluster Size:", clusters(i).size)
    }
    clusters
  }

  /**
   * Writes the modeling header to a CSV file
   */
  def writeHeader(filename: String): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File(filename)
      )
    )
    bw.write("gis_join,end_timestamp,is_centroid,time_ms,iterations,rmse,num_records\n")
    bw.close()
  }

  /**
   * Writes the modeling stats for a single model to a CSV file
   */
  def writeModelStats(filename: String, gisJoin: String, beginTimestamp: Long, isCentroid: String,
                      time: Long, iterations: Int, rmse: Double, numRecords: Long): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File(filename),
        true
      )
    )
    bw.write("%s,%d,%s,%d,%d,%f,%d\n".format(gisJoin, beginTimestamp, isCentroid, time, iterations, rmse, numRecords))
    bw.close()
  }

}
