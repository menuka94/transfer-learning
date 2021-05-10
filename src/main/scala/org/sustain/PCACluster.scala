package org.sustain

import scala.collection.mutable.ArrayBuffer

class PCACluster {

  var clusterId: Int = -1
  var centerGisJoin: String = ""
  var clusterGisJoins: ArrayBuffer[String] = ArrayBuffer[String]()

  /**
   * Overrides the toString method, for debugging model queues
   * @return String representation of PCACluster
   */
  override def toString: String = {
    "%d:%s:%s".format(clusterId, centerGisJoin, clusterGisJoins.toString())
  }
}
