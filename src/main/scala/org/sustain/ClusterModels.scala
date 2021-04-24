package org.sustain

import org.apache.spark.ml.regression.LinearRegression

class ClusterModels(sparkMasterC: String, mongoHostC: String, mongoPortC: String, databaseC: String,
                    collectionC: String, clusterIdC: Int, centroidEstimatorC: LinearRegression) {

  val sparkMaster: String = sparkMasterC
  val mongoUri: String = "mongodb://%s:%s/".format(mongoHostC, mongoPortC)
  val database: String = databaseC
  val collection: String = collectionC
  val clusterId: Int = clusterIdC
  val centroidEstimator: LinearRegression = centroidEstimatorC



}
