package org.sustain

import scala.collection.mutable.ArrayBuffer
import java.io.PrintWriter
import java.io.File

class Profiler {

  var tasks: ArrayBuffer[Task] = ArrayBuffer[Task]()

  def addTask(name: String): Unit = {
    this.tasks += new Task(name, System.currentTimeMillis())
  }

  def finishTask(name: String): Unit = {
    for (task <- this.tasks) {
      if (task.name == name) {
        task.finish()
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
