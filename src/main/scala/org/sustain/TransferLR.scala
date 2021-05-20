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

    writeClusterModelStats(clusterStatsCSVFilename, gisJoin, clusterId, end-begin, rmse, iterations)

    // <<< End Task for single cluster model's train() function
    profiler.finishTask(trainTaskId, System.currentTimeMillis())
  }

  /**
   * Function for testing single-model training
   */
  def testTrain(): Unit = {

    val conf: SparkConf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("Test Single Model Training")
      .set("spark.executor.cores", "8")
      .set("spark.executor.memory", "16G")
      .set("spark.mongodb.input.uri", "mongodb://lattice-100:27018/") // default mongos router
      .set("spark.mongodb.input.database", "sustaindb") // sustaindb
      .set("spark.mongodb.input.collection", "noaa_nam") // noaa_nam
      //.set("spark.mongodb.input.partitioner", "MongoShardedPartitioner")
      //.set("spark.mongodb.input.partitionerOptions.shardkey", "gis_join")

    val sparkSession: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()

    val regressionFeatures: Array[String] = Array(
      "relative_humidity_percent",
      "orography_surface_level_meters",
      "relative_humidity_percent",
      "pressure_pascal",
      "visibility_meters",
      "total_cloud_cover_percent",
      "10_metre_u_wind_component_meters_per_second",
      "10_metre_v_wind_component_meters_per_second"
    )
    val regressionLabel: String = "temp_surface_level_kelvin"
    val gisJoin: String = "G3100310"


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
      .withColumnRenamed(regressionLabel, "label")
      .filter(
        col("timestep") === 0 && col("gis_join") === gisJoin
      )


    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(regressionFeatures)
      .setOutputCol("features")

    mongoCollection = assembler.transform(mongoCollection)
      .select("gis_join", "features", "label")
      .persist()

    // Split into train/test sets
    val Array(train, test): Array[Dataset[Row]] = mongoCollection.randomSplit(Array(0.8, 0.2), 42)

    val numRecords: Long = mongoCollection.count()

    println("\n\nNUMBER OF ROWS: %d\n".format(numRecords))

    val linearRegression: LinearRegression = new LinearRegression()
      .setFitIntercept(true)
      .setLoss("squaredError")
      .setSolver("l-bfgs")
      .setRegParam(0.0)
      .setTol(0.001)
      .setMaxIter(100)
      .setEpsilon(1.35)
      .setElasticNetParam(0.0)
      .setStandardization(true)


    println("\n>>> BEGIN TS: %d\n".format(System.currentTimeMillis()))
    linearRegression.fit(train)
    println("\n>>> END TS: %d\n".format(System.currentTimeMillis()))

    sparkSession.close()
  }



  /**
   * Writes the modeling stats for a single model to a CSV file
   */
  def writeClusterModelStats(filename: String, gisJoin: String, clusterId: Int, time: Long, rmse: Double,
                             iterations: Int): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File(filename),
        true
      )
    )
    bw.write("%s,%d,%d,%d,%f\n".format(gisJoin, clusterId, time, iterations, rmse))
    bw.close()
  }

}
