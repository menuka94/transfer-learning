package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.col

import java.io.{BufferedWriter, File, FileWriter}

class SequentialTraining(sparkMasterC: String, mongoUriC: String, databaseC: String, collectionC: String,
                         gisJoinsC: Array[String], featuresC: Array[String], labelC: String) {

  val sparkMaster: String = sparkMasterC
  val mongoUri: String = mongoUriC
  val database: String = databaseC
  val collection: String = collectionC
  val gisJoins: Array[String] = gisJoinsC
  val features: Array[String] = featuresC
  val label: String = labelC

  def runNonTransferLearnedModels(): Unit = {

    val conf: SparkConf = new SparkConf()
      .setMaster(this.sparkMaster)
      .setAppName("Sequential, Non-TL LR Models")
      .set("spark.executor.cores", "8")
      .set("spark.executor.memory", "16G")
      .set("spark.mongodb.input.uri", this.mongoUri) // default mongos router
      .set("spark.mongodb.input.database", this.database) // sustaindb
      .set("spark.mongodb.input.collection", this.collection) // noaa_nam
      .set("spark.mongodb.input.readPreference", "secondary")

    val sparkSession: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()

    // Load in Dataset and persist it
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
      .withColumnRenamed(this.label, "label")
      .filter(col("timestep") === 0)

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(this.features)
      .setOutputCol("features")

    mongoCollection = assembler.transform(mongoCollection)
      .select("gis_join", "features", "label")

    // Sequentially train all models, without transfer-learning
    gisJoins.foreach(
      gisJoin => {

        // Filter down to just this GISJoin
        val gisJoinCollection = mongoCollection.filter(
          col("gis_join") === gisJoin
        ).persist()

        // Split Dataset into train/test sets
        val Array(train, test): Array[Dataset[Row]] = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)

        // Create basic Linear Regression Estimator
        val linearRegression: LinearRegression = new LinearRegression()
          .setFitIntercept(true)
          .setMaxIter(10)
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
        writeModelStats("no_tl_model_stats.csv", gisJoin, end-begin, iterations, rmse, numRecords)
      }
    )

    mongoCollection.unpersist()
  }

  def runTransferLearnedModels(): Unit = {

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
    bw.write("gis_join,time_ms,iterations,rmse,num_records\n")
    bw.close()
  }

  /**
   * Writes the modeling stats for a single model to a CSV file
   */
  def writeModelStats(filename: String, gisJoin: String, time: Long, iterations: Int, rmse: Double, numRecords: Long): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File(filename),
        true
      )
    )
    bw.write("%s,%d,%d,%f,%d\n".format(gisJoin, time, iterations, rmse, numRecords))
    bw.close()
  }

}
