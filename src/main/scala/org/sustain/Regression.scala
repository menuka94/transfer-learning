package org.sustain

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class Regression(gisJoinC: String, collectionC: Dataset[Row]) extends Thread {

  val gisJoin: String = gisJoinC
  val collection: Dataset[Row] = collectionC
  val REGRESSION_FEATURES: Array[String] = Array("year_month_day_hour")
  val REGRESSION_LABEL: String = "temp_surface_level_kelvin"
  val linearRegression: LinearRegression = new LinearRegression()

  def train() {
    println("\n\n>>> Fitting model for GISJoin " + gisJoin)

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
