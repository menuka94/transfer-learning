package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

class Regression(sparkSession: SparkSession, gisJoinC: String) extends Thread with Serializable {

  val gisJoin: String = gisJoinC
  val REGRESSION_FEATURES: Array[String] = Array("year_month_day_hour")
  val REGRESSION_LABEL: String = "temp_surface_level_kelvin"
  val linearRegression: LinearRegression = new LinearRegression()

  def train() {
    println("\n\n>>> Fitting model for GISJoin " + gisJoin)

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
    var gisJoinCollection: Dataset[Row] = collection.filter(col("gis_join") === gisJoin)
      .withColumnRenamed(REGRESSION_LABEL, "label")

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(REGRESSION_FEATURES)
      .setOutputCol("features")
    gisJoinCollection = assembler.transform(gisJoinCollection)

    // Split input into testing set and training set:
    // 80% training, 20% testing, with random seed of 42
    val Array(train, test): Array[Dataset[Row]] = gisJoinCollection.randomSplit(Array(0.8, 0.2), 42)

    // Create a linear regression model object and fit it to the training set
    val lrModel: LinearRegressionModel = linearRegression.fit(train)

    // Use the model on the testing set, and evaluate results
    val lrPredictions: DataFrame = lrModel.transform(test)
    val evaluator: RegressionEvaluator = new RegressionEvaluator().setMetricName("rmse")
    println("\n\n>>> TEST SET RMSE: " + evaluator.evaluate(lrPredictions))

  }

  override def run(): Unit = {
    train()
  }
}
