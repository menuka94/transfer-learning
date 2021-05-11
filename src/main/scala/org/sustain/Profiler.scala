package org.sustain

import scala.collection.mutable.ArrayBuffer
import java.io.PrintWriter
import java.io.File

class Profiler {

  var tasks: ArrayBuffer[Task] = ArrayBuffer[Task]()
  val jobBeginTimestampMs: Long = System.currentTimeMillis()
  val printWriter: PrintWriter = new PrintWriter(new File("profiler_logs.txt"))

  def addTask(name: String): Int = {
    this.synchronized {
      this.tasks += new Task(name, this.jobBeginTimestampMs)
      printWriter.write("START TASK %d: NAME: %s\n".format(this.tasks.length-1, name))
      this.tasks.length - 1
    }
  }

  def finishTask(id: Int): Boolean = {
    this.finishTask(id, System.currentTimeMillis())
  }

  def finishTask(id: Int, timestamp: Long): Boolean = {
    this.synchronized {
      if (this.tasks(id).endTimestamp == -1) {
        this.tasks(id).finish(timestamp)
        printWriter.write("FINISH TASK %d: %s\n".format(id, this.tasks(id)))
        true
      } else {
        printWriter.write("ERROR TASK %d: %s\n".format(id, this.tasks(id)))
        false
      }
    }
  }

  def writeToFile(filename: String): Unit = {
    val pw: PrintWriter = new PrintWriter(new File(filename))
    pw.write("Total tasks: %d\n".format(this.tasks.length))
    pw.write("name,begin,end,time_seconds\n")
    for (task <- this.tasks) {
      pw.write(task.toString + "\n")
    }
    pw.close()
  }

  def close(): Unit = {
    this.synchronized {
      this.printWriter.close()
    }
  }

}
