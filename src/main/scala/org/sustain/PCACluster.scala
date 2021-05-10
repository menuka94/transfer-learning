package org.sustain

class PCACluster {

  var clusterId: Int = -1
  var centerGisJoin: String = null
  var clusterGisJoins: List[String] = List[String]()

  /**
   * Overrides the toString method, for debugging model queues
   * @return String representation of PCACluster
   */
  override def toString: String = {
    "%d:%s:%s".format(clusterId, centerGisJoin, clusterGisJoins.toString())
  }
}
