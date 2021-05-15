package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.col
import com.mongodb.spark.config._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}

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
    //val trainTaskId: Int = this.profiler.addTask(trainTaskName)
    println("\n\n" + trainTaskName)

    val readConfig: ReadConfig = ReadConfig(
      Map(
        "uri" -> this.mongoUri,
        "database" -> this.database,
        "collection" -> this.collection
      ), Some(ReadConfig(this.sparkSession))
    )

    // Load in Dataset; reduce it down to rows for this GISJoin at timestep 0; persist it for multiple operations
    val mongoCollection: Dataset[Row] = MongoSpark.load(this.sparkSession, readConfig)
      .select("gis_join", "pressure_pascal", "timestep", "temp_surface_level_kelvin")
      .withColumnRenamed(this.label, "label")
      .filter(col("gis_join") === this.gisJoin && col("timestep") === 0)
      .persist()

    // Split Dataset into train/test sets
    val Array(train, test): Array[Dataset[Row]] = mongoCollection.randomSplit(Array(0.8, 0.2), 42)

    // Create a ML Pipeline using VectorAssembler and Linear Regression
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(this.features)
      .setOutputCol("features")
    val linearRegression: LinearRegression = new LinearRegression()
      .setFitIntercept(true)
      .setMaxIter(10)
      .setLoss("squaredError")
      .setSolver("l-bfgs")
      .setStandardization(true)
    val pipeline = new Pipeline()
      .setStages(Array(assembler, linearRegression))

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // With 3 values for tolerance, 3 values for regularization param, and 3 values for epsilon,
    // this grid will have 3 x 3 x 3 = 27 parameter settings for CrossValidator to choose from.
    val paramGrid: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(linearRegression.tol, Array(0.1, 0.01, 0.001))
      .addGrid(linearRegression.regParam, Array(0.1, 0.3, 0.5))
      .addGrid(linearRegression.epsilon, Array(1.1, 1.35, 1.5))
      .build()

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val crossValidator: CrossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)     // Use 3+ in practice
      .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

    // Run cross-validation, and choose the best set of parameters.
    val crossValidatorModel: CrossValidatorModel = crossValidator.fit(train)
    println("\n\n>>> Params: %s\n".format(crossValidatorModel.params))

    // Make predictions on the testing Dataset, evaluate performance
    val predictions: Dataset[Row] = crossValidatorModel.transform(test)
    val evaluator: RegressionEvaluator = new RegressionEvaluator().setMetricName("rmse")
    println("\n\n>>> Test set RMSE for " + gisJoin + ": " + evaluator.evaluate(predictions))


    // Unpersist Dataset to free up space
    mongoCollection.unpersist()
    //this.profiler.finishTask(trainTaskId, System.currentTimeMillis())
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
