package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.col
import com.mongodb.spark.config._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}

import java.io.{BufferedWriter, File, FileWriter}

class CentroidModel(sparkMasterC: String, mongoUriC: String, databaseC: String, collectionC: String,
                    labelC: String, featuresC: Array[String], gisJoinC: String, clusterIdC: Int,
                    sparkSessionC: SparkSession, profilerC: Profiler, statsCSVFilename: String)
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
  val centroidModelStatsCSV: String = statsCSVFilename

  /**
   * Launched by the thread.start()
   */
  override def run(): Unit = {
    // >>> Begin Task for centroid model's run() function
    val trainTaskName: String = "CentroidModel;run();gisJoin=%s;clusterId=%d".format(this.gisJoin, this.clusterId)
    val trainTaskId: Int = this.profiler.addTask(trainTaskName)
    println("\n\n" + trainTaskName)

    val readConfig: ReadConfig = ReadConfig(
      Map(
        "uri" -> this.mongoUri,
        "database" -> this.database,
        "collection" -> this.collection
      ), Some(ReadConfig(this.sparkSession))
    )

    // Load in Dataset; reduce it down to rows for this GISJoin at timestep 0; persist it for multiple operations
    var mongoCollection: Dataset[Row] = MongoSpark.load(this.sparkSession, readConfig)
      .select("gis_join", "relative_humidity_percent", "timestep", "temp_surface_level_kelvin")
      .withColumnRenamed(this.label, "label")
      .filter(col("gis_join") === this.gisJoin && col("timestep") === 0)

    // Assemble features into vector column and persist Dataset
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(this.features)
      .setOutputCol("features")
    mongoCollection = assembler.transform(mongoCollection).persist()
    val numRecords: Long = mongoCollection.count()

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
    this.linearRegression = bestLRModel.parent.asInstanceOf[LinearRegression]
    val bestLRRegParam: Double = bestLRModel.getRegParam
    val bestLRTol: Double = bestLRModel.getTol
    val bestLREpsilon: Double = bestLRModel.getEpsilon
    val bestLRIterations: Int = bestLRModel.summary.totalIterations

    // Make predictions on the testing Dataset, evaluate performance
    val predictions: Dataset[Row] = bestLRModel.transform(test)
    val rmse: Double = evaluator.evaluate(predictions)
    println("\n\n>>> Test set RMSE for %s: %f\n".format(this.gisJoin, rmse))

    // Write stats to CSV
    writeCentroidModelStats(this.statsCSVFilename, this.gisJoin, this.clusterId, numRecords, end-begin, rmse, bestLRIterations,
      bestLRRegParam, bestLRTol, bestLREpsilon)

    // Unpersist Dataset to free up space
    mongoCollection.unpersist()

    // <<< End Task for cluster's run() function
    this.profiler.finishTask(trainTaskId, System.currentTimeMillis())
  }

  def writeCentroidModelStats(filename: String, gisJoin: String, clusterId: Int, numberRecords: Long, time: Long, rmse: Double,
                              iterations: Int, bestRegParam: Double, bestTol: Double, bestEpsilon: Double): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File(filename),
        true
      )
    )
    bw.write("%s,%d,%d,%d,%f,%d,%f,%f,%f\n".format(gisJoin, clusterId, numberRecords, time, rmse, iterations, bestRegParam, bestTol, bestEpsilon))
    bw.close()
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
