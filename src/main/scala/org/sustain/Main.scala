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
  val PCA_FEATURES: Array[String] = Array("mean_sea_level_pressure_pascal",
    "surface_pressure_surface_level_pascal",
    "orography_surface_level_meters",
    "temp_surface_level_kelvin",
    "2_metre_temp_kelvin",
    "2_metre_dewpoint_temp_kelvin",
    "relative_humidity_percent",
    "10_metre_u_wind_component_meters_per_second",
    "10_metre_v_wind_component_meters_per_second",
    "total_precipitation_kg_per_squared_meter",
    "water_convection_precipitation_kg_per_squared_meter",
    "soil_temperature_kelvin",
    "pressure_pascal",
    "visibility_meters",
    "precipitable_water_kg_per_squared_meter",
    "total_cloud_cover_percent",
    "snow_depth_meters",
    "ice_cover_binary")
  val CLUSTERING_FEATURES: Array[String] = Array("avg_pc_0", "avg_pc_1", "avg_pc_2", "avg_pc_3", "avg_pc_4", "avg_pc_5")
  val CLUSTERING_K: Int = 56 // sqrt(3192) = 56
  val REGRESSION_FEATURES: Array[String] = Array(
    "relative_humidity_percent",
    "orography_surface_level_meters",
    "relative_humidity_percent",
    "10_metre_u_wind_component_meters_per_second",
    "pressure_pascal",
    "visibility_meters",
    "total_cloud_cover_percent",
    "10_metre_u_wind_component_meters_per_second",
    "10_metre_v_wind_component_meters_per_second"
  )
  val REGRESSION_LABEL: String = "temp_surface_level_kelvin"
  val PROFILE_OUTPUT: String = "experiment_profile.csv"
  val CENTROID_STATS_CSV: String = "centroid_stats.csv"
  val CLUSTER_MODEL_STATS_CSV: String = "cluster_model_stats.csv"
  val SEQUENTIAL_STATS_CSV: String = "sequential_stats.csv"

  /* Entrypoint for the application */
  def main(args: Array[String]): Unit = {

    System.setProperty("mongodb.keep_alive_ms", "100000")
    val experiment: Experiment = new Experiment()
    val seqTraining: SequentialTraining = new SequentialTraining(SPARK_MASTER, "mongodb://lattice-100:27018",
      MONGO_DB, MONGO_COLLECTION, experiment.loadGisJoins("experiment_data/job_profiles/clusters_pck6.csv"),
      REGRESSION_FEATURES, REGRESSION_LABEL)

    seqTraining.run()

    //
    // experiment.runSingleModelTest()

//    println("\n\n>>> Starting nanosecond timer\n")
//    experiment.pcaClustering(SPARK_MASTER, APP_NAME, MONGO_ROUTER_HOSTS, MONGO_PORT, MONGO_DB,
//      MONGO_COLLECTION, CLUSTERING_FEATURES, CLUSTERING_K, PCA_FEATURES)

//    experiment.transferLearning(SPARK_MASTER, APP_NAME, MONGO_ROUTER_HOSTS, MONGO_PORT, MONGO_DB,
//      MONGO_COLLECTION, REGRESSION_FEATURES, REGRESSION_LABEL, PROFILE_OUTPUT, CENTROID_STATS_CSV,
//      CLUSTER_MODEL_STATS_CSV, experiment.loadClusters("experiment_data/job_profiles/clusters.csv", 56))


//    experiment.sequentialTraining(SPARK_MASTER, APP_NAME, MONGO_ROUTER_HOSTS, MONGO_PORT, MONGO_DB, MONGO_COLLECTION,
//      REGRESSION_FEATURES, REGRESSION_LABEL, SEQUENTIAL_STATS_CSV, PROFILE_OUTPUT, experiment.loadGisJoins("experiment_data/job_profiles/clusters.csv"))

  }

  /**
   * Records the wall-clock time that a block of code takes to run.
   * Can be wrapped around function calls or multiple statements.
   * @param block The block of code we are wrapping
   * @tparam R The return value of the block of code (last line in the block)
   * @return Returns whatever the last line in the code block returned.
   */
  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("\n\n>>> Elapsed time: " + ( (t1 - t0) / 10E8 )  + " seconds") // Convert nanoseconds to seconds
    result
  }


}
