package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

class ClusterLRModels(sparkMasterC: String, mongoHostC: String, mongoPortC: String, databaseC: String,
                      collectionC: String, clusterIdC: Int, gisJoinsC: Array[String],
                      centroidEstimatorC: LinearRegression, featuresC: Array[String], labelC: String)
                      extends Thread with Serializable {

  val sparkMaster: String = sparkMasterC
  val mongoUri: String = "mongodb://%s:%s/".format(mongoHostC, mongoPortC)
  val database: String = databaseC
  val collection: String = collectionC
  val clusterId: Int = clusterIdC
  val gisJoins: Array[String] = gisJoinsC
  val centroidEstimator: LinearRegression = centroidEstimatorC
  val features: Array[String] = featuresC
  val label: String = labelC

  /**
   * Launched by the thread.start()
   */
  override def run(): Unit = {
    println("\n\n>>> Fitting models for cluster " + clusterId)

    val conf: SparkConf = new SparkConf()
      .setMaster(this.sparkMaster)
      .setAppName("Cluster %d models, MongoS [%s]".format(this.clusterId, mongoHostC))
      .set("spark.executor.cores", "2")
      .set("spark.executor.memory", "1G")
      .set("spark.mongodb.input.uri", this.mongoUri)
      .set("spark.mongodb.input.database", this.database)
      .set("spark.mongodb.input.collection", this.collection)

    // Create the SparkSession and ReadConfig
    val sparkSession: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate() // For the $()-referenced columns

    /* Read collection into a DataSet[Row], dropping null rows
          +--------+-------------------+-------------------------+
          |gis_join|year_month_day_hour|temp_surface_level_kelvin|
          +--------+-------------------+-------------------------+
          |G4804230|         2010010100|        281.4640808105469|
          |G5600390|         2010010100|        265.2140808105469|
          |G1701150|         2010010100|        265.7140808105469|
          |G0601030|         2010010100|        282.9640808105469|
          |G3701230|         2010010100|        279.2140808105469|
          |G3700690|         2010010100|        280.8390808105469|
          |G3701070|         2010010100|        280.9640808105469|
          |G4803630|         2010010100|        275.7140808105469|
          |G5108200|         2010010100|        273.4640808105469|
          |G4801170|         2010010100|        269.3390808105469|
          +--------+-------------------+-------------------------+
         */
    var mongoCollection: Dataset[Row] = MongoSpark.load(sparkSession)
    mongoCollection = mongoCollection.select("gis_join", "year_month_day_hour", "temp_surface_level_kelvin")
      .na.drop()
    mongoCollection.persist() // Persist collection for reuse

    // Iterate over all gisJoins in this collection, build models for each from persisted collection
    gisJoins.foreach(
      gisJoin => {

        // Filter the data down to just entries for a single GISJoin
        var gisJoinCollection: Dataset[Row] = mongoCollection.filter(col("gis_join") === gisJoin)
          .withColumnRenamed(this.label, "label")

        val assembler: VectorAssembler = new VectorAssembler()
          .setInputCols(this.features)
          .setOutputCol("features")
        gisJoinCollection = assembler.transform(gisJoinCollection)

        // Split input into testing set and training set:
        // 80% training, 20% testing, with random seed of 42
        val Array(train, test): Array[Dataset[Row]] = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)

        // Copy the hyper-parameters from the already-trained centroid model for this cluster
        val linearRegression: LinearRegression = centroidEstimator.copy(new ParamMap())

        // Create a linear regression model object and fit it to the training set
        val lrModel: LinearRegressionModel = linearRegression.fit(train)

        // Use the model on the testing set, and evaluate results
        val lrPredictions: DataFrame = lrModel.transform(test)
        val evaluator: RegressionEvaluator = new RegressionEvaluator().setMetricName("rmse")
        println("\n\n>>> Test set RMSE for " + gisJoin + ": " + evaluator.evaluate(lrPredictions))

      }
    )

    mongoCollection.unpersist()
  }

  /**
   * Overrides the toString method, for debugging model queues
   * @return String representation of Regression
   */
  override def toString: String = {
    var retVal: String = "Cluster ID: [%d]: [ ".format(clusterId)
    gisJoins.foreach(gisJoin => retVal += gisJoin + " ")
    retVal += "]\n"
    retVal
  }

}
