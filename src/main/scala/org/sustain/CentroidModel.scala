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


  /**
   * Launched by the thread.start(), executes train()
   */
  override def run(): Unit = {
    println("\n\n>>> Fitting centroid model for GISJoin " + gisJoin)

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

    /* Read collection into a DataSet[Row], dropping null rows
      +--------+-------------------+--------+-------------------------+
      |gis_join|year_month_day_hour|timestep|temp_surface_level_kelvin|
      +--------+-------------------+--------+-------------------------+
      |G4804230|         2010010100|       0|        281.4640808105469|
      |G5600390|         2010010100|       0|        265.2140808105469|
      |G1701150|         2010010100|       0|        265.7140808105469|
      |G0601030|         2010010100|       0|        282.9640808105469|
      |G3701230|         2010010100|       0|        279.2140808105469|
      |G3700690|         2010010100|       0|        280.8390808105469|
      |G3701070|         2010010100|       0|        280.9640808105469|
      |G4803630|         2010010100|       0|        275.7140808105469|
      |G5108200|         2010010100|       0|        273.4640808105469|
      |G4801170|         2010010100|       0|        269.3390808105469|
      +--------+-------------------+--------+-------------------------+
     */
    var collection: Dataset[Row] = MongoSpark.load(sparkSession)
    collection = collection.select("gis_join", "year_month_day_hour", "timestep", "temp_surface_level_kelvin")
      .na.drop()

    // Filter the data down to just entries for a single GISJoin
    var gisJoinCollection: Dataset[Row] = collection.filter(col("gis_join") === this.gisJoin)
      .withColumnRenamed(this.label, "label")

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(this.features)
      .setOutputCol("features")
    gisJoinCollection = assembler.transform(gisJoinCollection)

    // Split input into testing set and training set:
    // 80% training, 20% testing, with random seed of 42
    val Array(train, test): Array[Dataset[Row]] = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)

    // Create a linear regression model object and fit it to the training set
    val lrModel: LinearRegressionModel = this.linearRegression.fit(train)

    // Use the model on the testing set, and evaluate results
    val lrPredictions: DataFrame = lrModel.transform(test)
    val evaluator: RegressionEvaluator = new RegressionEvaluator().setMetricName("rmse")
    println("\n\n>>> Test set RMSE for " + this.gisJoin + ": " + evaluator.evaluate(lrPredictions))
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
