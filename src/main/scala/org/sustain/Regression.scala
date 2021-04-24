package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.ParamMap

class Regression(gisJoinC: String, clusterIdC: Int, mongoHostC: String) extends Thread with Serializable with Ordered[Regression] {

  val gisJoin: String = gisJoinC
  val clusterId: Int = clusterIdC
  val REGRESSION_FEATURES: Array[String] = Array("year_month_day_hour")
  val REGRESSION_LABEL: String = "temp_surface_level_kelvin"
  var linearRegression: LinearRegression = new LinearRegression()
  val SPARK_MASTER: String = "spark://lattice-100:8079"
  val APP_NAME: String = "Transfer Learning"
  val MONGO_HOST: String = mongoHostC
  val MONGO_PORT: String = "27018"
  val MONGO_URI: String = "mongodb://" + MONGO_HOST + ":" + MONGO_PORT +"/"
  val MONGO_DB: String = "sustaindb"
  val MONGO_COLLECTION: String = "noaa_nam"

  def trainCentroid(): Unit = {

  }

  def train() {


  }

  /**
   * Launched by the thread, executes train()
   */
  override def run(): Unit = {
    train()
  }

  /**
   * Allows ordering of Regression objects, sorted by ascending cluster id which the
   * gisjoin belongs to.
   * @param that The other Regression instance we are comparing ourselves to
   * @return 0 if the cluster ids are equal, 1 if our cluster id is greater than the other Regression instance, and we
   *         should come after "that", and -1 if our cluster id is less than the other Regression instance, and we
   *         should come before "that".
   */
  override def compare(that: Regression): Int = {
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
