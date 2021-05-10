package org.sustain

import scala.collection.mutable.ArrayBuffer
import java.io.PrintWriter
import java.io.File

class Profiler {

  var tasks: ArrayBuffer[Task] = ArrayBuffer[Task]()
  val jobBeginTimestampMs: Long = System.currentTimeMillis()

  def addTask(name: String): Unit = {
    this.synchronized {
      this.tasks += new Task(name, this.jobBeginTimestampMs)
    }
  }

  def finishTask(name: String): Unit = {
    this.synchronized {
      for (task <- this.tasks) {
        if (task.name == name) {
          task.finish()
        }
      }
    }
  }

  def writeToFile(filename: String): Unit = {
    val pw: PrintWriter = new PrintWriter(new File(filename))
    pw.write("name,begin,end,time_seconds\n")
    for (task <- this.tasks) {
      pw.write(task.toString + "\n")
    }
    pw.close()
  }

}
