package org.sustain

import scala.collection.mutable.ArrayBuffer

class PCACluster extends Ordered[PCACluster] {

  var clusterId: Int = -1
  var centerGisJoin: String = ""
  var clusterGisJoins: ArrayBuffer[String] = ArrayBuffer[String]()

  /**
   * Allows ordering of PCACluster objects, sorted by ascending cluster id.
   * @param that The other PCACluster instance we are comparing ourselves to
   * @return 0 if the cluster ids are equal, 1 if our cluster id is greater than the other PCACluster instance, and we
   *         should come after "that", and -1 if our cluster id is less than the other PCACluster instance, and we
   *         should come before "that".
   */
  override def compare(that: PCACluster): Int = {
    if (this.clusterId == that.clusterId)
      0
    else if (this.clusterId > that.clusterId)
      1
    else
      -1
  }

  /**
   * Overrides the toString method, for debugging model queues
   * @return String representation of PCACluster
   */
  override def toString: String = {
    "%d:%s:%s\n".format(clusterId, centerGisJoin, clusterGisJoins.toString())
  }
}
