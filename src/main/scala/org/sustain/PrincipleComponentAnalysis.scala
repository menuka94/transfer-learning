package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel, PCA, PCAModel, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

class PrincipleComponentAnalysis {

  def runPCA(sparkMaster: String, appName: String, mongosRouters: Array[String], mongoPort: String,
             database: String, collection: String, pcaFeatures: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName(appName)
      .set("spark.executor.cores", "8")
      .set("spark.executor.memory", "20G")
      .set("spark.mongodb.input.uri", "mongodb://%s:%s/".format(mongosRouters(0), mongoPort))
      .set("spark.mongodb.input.database", database)
      .set("spark.mongodb.input.collection", collection)
      .set("mongodb.keep_alive_ms", "100000") // Important! Default is 5000ms, and stream will prematurely close

    // Create the SparkSession and ReadConfig
    val sparkConnector: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate() // For the $()-referenced columns

    import sparkConnector.implicits._

    val mongoCollection: Dataset[Row] = MongoSpark.load(sparkConnector)

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(pcaFeatures)
      .setOutputCol("features")
    val withFeaturesAssembled: Dataset[Row] = assembler.transform(mongoCollection)

    val minMaxScaler: MinMaxScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("normalized_features")
    val minMaxScalerModel: MinMaxScalerModel = minMaxScaler.fit(withFeaturesAssembled)
    var normalizedFeatures: Dataset[Row] = minMaxScalerModel.transform(withFeaturesAssembled)
    normalizedFeatures = normalizedFeatures.drop("features")
    normalizedFeatures = normalizedFeatures.withColumnRenamed("normalized_features", "features")

    val pca: PCAModel = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(6)
      .fit(normalizedFeatures)

    val pc: DenseMatrix = pca.pc
    val pcaDF: Dataset[Row] = pca.transform(normalizedFeatures).select("gis_join", "features", "pcaFeatures");
    // val requiredNoOfPCs: Int = getNoPrincipalComponentsByVariance(pca, 0.95); // 6
    pcaDF.show(100)
  }

  def getNoPrincipalComponentsByVariance(pca: PCAModel, targetVariance: Double): Int = {
    var n: Int = -1
    var varianceSum = 0.0
    val explainedVariance: DenseVector = pca.explainedVariance
    explainedVariance.foreachActive((index, variance) => {
      n = index + 1
      if (n >= pca.getK) {
        return n
      }
      varianceSum += variance
      if (varianceSum >= targetVariance) {
        return n
      }
    })

    pca.getK
  }

  /*
  def getNoOfPrincipalComponentsByVariance(pca: PCAModel, double targetVariance) {
    int n;
    double varianceSum = 0.0;
    DenseVector explainedVariance = pca.explainedVariance();
    Iterator<Tuple2<Object, Object>> iterator = explainedVariance.iterator();
    while (iterator.hasNext()) {
      Tuple2<Object, Object> next = iterator.next();
      n = Integer.parseInt(next._1().toString()) + 1;
      if (n >= pca.getK()) {
        break;
      }
      varianceSum += Double.parseDouble(next._2().toString());
      if (varianceSum >= targetVariance) {
        return n;
      }
    }

    return pca.getK();
  }

   */

}
