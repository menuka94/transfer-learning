/* -----------------------------------------------
 * Main.scala -
 *
 * Description:
 *    Provides a demonstration of the Spark capabilities.
 *    Guide for this project taken directly from MongoDB docs:
 *    https://docs.mongodb.com/spark-connector/master/scala-api
 *
 *  Author:
 *    Caleb Carlson
 *
 * ----------------------------------------------- */
package org.sustain

object Main {

  /* Global Variables */
  val SPARK_MASTER: String = "spark://lattice-100:8079"
  val APP_NAME: String = "Transfer Learning"
  val MONGO_ROUTER_HOSTS: Array[String] = Array("lattice-100", "lattice-101", "lattice-102", "lattice-103", "lattice-104")
  val MONGO_PORT: String = "27018"
  val MONGO_DB: String = "sustaindb"
  val MONGO_COLLECTION: String = "noaa_nam"
  val CLUSTERING_FEATURES: Array[String] = Array("temp_surface_level_kelvin")
  val CLUSTERING_TIMESTEP: Long = 0
  val CLUSTERING_K: Int = 56 // sqrt(3192) = 56
  val CLUSTERING_YEAR_MONTH_DAY_HOUR: Long = 2010010100
  val REGRESSION_FEATURES: Array[String] = Array("year_month_day_hour")
  val REGRESSION_LABEL: String = "temp_surface_level_kelvin"

  /* Entrypoint for the application */
  def main(args: Array[String]): Unit = {

    val experiment: Experiment = new Experiment()
    println("\n\n>>> Starting nanosecond timer\n")
    time { experiment.transferLearning(SPARK_MASTER, APP_NAME, MONGO_ROUTER_HOSTS, MONGO_PORT, MONGO_DB,
      MONGO_COLLECTION, CLUSTERING_FEATURES, CLUSTERING_YEAR_MONTH_DAY_HOUR, CLUSTERING_TIMESTEP, CLUSTERING_K,
      REGRESSION_FEATURES, REGRESSION_LABEL) }

  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("\n\n>>> Elapsed time: " + ( (t1 - t0) / 10E8 )  + " seconds") // Convert nanoseconds to seconds
    result
  }


}
