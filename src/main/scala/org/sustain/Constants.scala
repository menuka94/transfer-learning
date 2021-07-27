package org.sustain

object Constants {
  val DB_HOST: String = sys.env("DB_HOST")
  val DB_PORT: Int = sys.env("DB_PORT").toInt
  val SPARK_MASTER: String = sys.env("SPARK_MASTER")
  val KUBERNETES_SPARK_MASTER: String = sys.env("KUBERNETES_SPARK_MASTER")
  val USE_KUBERNETES: Boolean = sys.env("USE_KUBERNETES").toBoolean
  val SPARK_INITIAL_EXECUTORS: Int = sys.env("SPARK_INITIAL_EXECUTORS").toInt
  val SPARK_MIN_EXECUTORS: Int = sys.env("SPARK_MIN_EXECUTORS").toInt
  val SPARK_MAX_EXECUTORS: Int = sys.env("SPARK_MAX_EXECUTORS").toInt
  val SPARK_EXECUTOR_MEMORY: String = sys.env("SPARK_EXECUTOR_MEMORY")
  val SPARK_BACKLOG_TIMEOUT: String = sys.env("SPARK_BACKLOG_TIMEOUT")
  val SPARK_IDLE_TIMEOUT: String = sys.env("SPARK_IDLE_TIMEOUT")
  val SPARK_DOCKER_IMAGE: String = sys.env("SPARK_DOCKER_IMAGE")

  val GIS_JOIN = "gis_join"
}
