package org.sustain

class Task(nameC: String) {

  val name: String = nameC
  val beginTimestamp: Long = System.nanoTime()
  var endTimestamp: Long = -1L

  def finish(): Unit = {
    this.endTimestamp = System.nanoTime()
  }

  def timeTaken(): Long = {
    this.endTimestamp - this.beginTimestamp
  }

  /**
   * Overrides the toString method, for debugging model queues
   * @return String representation of Task
   */
  override def toString: String = {
    "%s,%d,%d,%d".format(this.name, this.beginTimestamp, this.endTimestamp, this.timeTaken())
  }

}
