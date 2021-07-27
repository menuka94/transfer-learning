package org.sustain.util

import java.io.{File, FileWriter, PrintWriter}
import java.time.LocalDateTime

object Logger {
  val logFile: String = System.getenv("HOME") + "/sustain-transfer-learning.log"
  val pw: PrintWriter = new PrintWriter(new FileWriter(new File(logFile), true))
  def log(message: String) {
    val log = LocalDateTime.now() + ": " + message
    println(log)
    pw.write(log + "\n")
    pw.flush()
  }
}
