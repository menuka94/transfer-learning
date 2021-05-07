package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel, PCA, PCAModel, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

class PrincipleComponentAnalysis {

  def runPCA(sparkConnector: SparkSession, pcaFeatures: Array[String]): Dataset[Row] = {

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
    pcaDF
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
