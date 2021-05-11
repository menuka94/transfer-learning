package org.sustain

class Task(nameC: String, jobBeginTimestampMs: Long) {

  val name: String = nameC
  val beginTimestamp: Long = System.currentTimeMillis() - jobBeginTimestampMs
  var endTimestamp: Long = -1L

  def finish(timestamp: Long): Unit = {
    this.endTimestamp = timestamp - jobBeginTimestampMs
  }

  def timeTakenSec(): Double = {
    (this.endTimestamp - this.beginTimestamp) / 1000.0
  }

  /**
   * Overrides the toString method, for debugging model queues
   * @return String representation of Task
   */
  override def toString: String = {
    "%s,%d,%d,%.2f".format(this.name, this.beginTimestamp, this.endTimestamp, this.timeTakenSec())
  }

}
