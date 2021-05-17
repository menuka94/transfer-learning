package org.sustain

class SequentialTraining(sparkMasterC: String, mongoUriC: String, databaseC: String, collectionC: String,
                         gisJoinsC: Array[String], featuresC: Array[String], labelC: String) {

  val sparkMaster: String = sparkMasterC
  val mongoUri: String = mongoUriC
  val database: String = databaseC
  val collection: String = collectionC


  def run(): Unit = {

  }

}
