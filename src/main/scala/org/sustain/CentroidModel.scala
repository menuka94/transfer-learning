package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.col

class CentroidModel(sparkMasterC: String, mongoHostC: String, mongoPortC: String, databaseC: String,
                    collectionC: String, labelC: String, featuresC: Array[String], gisJoinC: String, clusterIdC: Int)
                    extends Thread with Serializable with Ordered[CentroidModel] {

  val linearRegression: LinearRegression = new LinearRegression()
  val sparkMaster: String = sparkMasterC
  val mongoUri: String = "mongodb://%s:%s/".format(mongoHostC, mongoPortC)
  val database: String = databaseC
  val collection: String = collectionC
  val label: String = labelC
  val features: Array[String] = featuresC
  val gisJoin: String = gisJoinC
  val clusterId: Int = clusterIdC
  //var mongoCollection: Dataset[Row] = mongoCollectionC

  /**
   * Launched by the thread.start()
   */
  override def run(): Unit = {
    val profiler: Profiler = new Profiler()
    println("\n\n>>> Fitting centroid model for GISJoin " + gisJoin + ", cluster " + clusterId + ", mongos " + mongoHostC)
    val taskName: String = "CentroidModel run() for GISJoin: %s, Cluster: %d, MongoS: %s".format(gisJoin, clusterId, mongoHostC)
    profiler.addTask(taskName)


    val conf: SparkConf = new SparkConf()
      .setMaster(this.sparkMaster)
      .setAppName("Centroid Model for GISJoin [%s], Cluster [%d], MongoS [%s]".format(this.gisJoin, this.clusterId, mongoHostC))
      .set("spark.executor.cores", "4")
      .set("spark.executor.memory", "8G")
      .set("spark.mongodb.input.uri", this.mongoUri)
      .set("spark.mongodb.input.database", this.database)
      .set("spark.mongodb.input.collection", this.collection)

    // Create the SparkSession and ReadConfig
    val sparkSession: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate() // For the $()-referenced columns


    /* Read collection into a DataSet[Row], dropping null rows, filter by this GISJoin, and timestep 0, and rename
       the label column to "label"
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
    var mongoCollection: Dataset[Row] = MongoSpark.load(sparkSession)
    mongoCollection = mongoCollection.select("gis_join", "year_month_day_hour", "timestep", "temp_surface_level_kelvin")
      .na.drop().filter(
      col("gis_join") === this.gisJoin && col("timestep") === 0
    ).withColumnRenamed(this.label, "label")
    //mongoCollection.show(10)

    /* Assemble into features
      +--------+-------------------+--------+------------------+------------+
      |gis_join|year_month_day_hour|timestep|             label|    features|
      +--------+-------------------+--------+------------------+------------+
      |G4801730|         2010011000|       0|272.89988708496094|[2.010011E9]|
      |G4801730|         2010011000|       0|272.89988708496094|[2.010011E9]|
      |G4801730|         2010011000|       0|272.89988708496094|[2.010011E9]|
      |G4801730|         2010011000|       0|272.52488708496094|[2.010011E9]|
      |G4801730|         2010011000|       0|273.89988708496094|[2.010011E9]|
      |G4801730|         2010011000|       0|272.77488708496094|[2.010011E9]|
      |G4801730|         2010011000|       0|272.89988708496094|[2.010011E9]|
      |G4801730|         2010011000|       0|274.14988708496094|[2.010011E9]|
      |G4801730|         2010011000|       0|273.14988708496094|[2.010011E9]|
      |G4801730|         2010011000|       0|274.14988708496094|[2.010011E9]|
      |G4801730|         2010011000|       0|273.27488708496094|[2.010011E9]|
      +--------+-------------------+--------+------------------+------------+
     */
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(this.features)
      .setOutputCol("features")
    mongoCollection = assembler.transform(mongoCollection)
    //mongoCollection.show(11)

    // Split input into testing set and training set:
    // 80% training, 20% testing, with random seed of 42
    val Array(train, test): Array[Dataset[Row]] = mongoCollection.randomSplit(Array(0.8, 0.2), 42)

    // Create a linear regression model object and fit it to the training set
    val lrModel: LinearRegressionModel = this.linearRegression.fit(train)

    // Use the model on the testing set, and evaluate results
    val lrPredictions: DataFrame = lrModel.transform(test)
    val evaluator: RegressionEvaluator = new RegressionEvaluator().setMetricName("rmse")
    println("\n\n>>> Test set RMSE for " + this.gisJoin + ": " + evaluator.evaluate(lrPredictions))

    profiler.finishTask(taskName)
    profiler.writeToFile("transfer_learning_profile.csv")
    sparkSession.close()
  }

  /**
   * Allows ordering of CentroidModel objects, sorted by ascending cluster id which the GISJoin belongs to.
   * @param that The other Regression instance we are comparing ourselves to
   * @return 0 if the cluster ids are equal, 1 if our cluster id is greater than the other Regression instance, and we
   *         should come after "that", and -1 if our cluster id is less than the other Regression instance, and we
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
