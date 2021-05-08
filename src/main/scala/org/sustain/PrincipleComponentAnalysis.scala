package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel, PCA, PCAModel, VectorAssembler, VectorDisassembler}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

class PrincipleComponentAnalysis {

  def runPCA(spark: SparkSession, pcaFeatures: Array[String]): Dataset[Row] = {

    val mongoCollection: Dataset[Row] = MongoSpark.load(spark)

    // Assemble all requested features into a Column of type Vector
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(pcaFeatures)
      .setOutputCol("features")
    val withFeaturesAssembled: Dataset[Row] = assembler.transform(mongoCollection)

    // Normalize the features, and replace the features column with those normalized values
    val minMaxScaler: MinMaxScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("normalized_features")
    val minMaxScalerModel: MinMaxScalerModel = minMaxScaler.fit(withFeaturesAssembled)
    var normalizedFeatures: Dataset[Row] = minMaxScalerModel.transform(withFeaturesAssembled)
    normalizedFeatures = normalizedFeatures.drop("features")
    normalizedFeatures = normalizedFeatures.withColumnRenamed("normalized_features", "features")

    // Run PCA on the normalized features
    val pca: PCAModel = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(6)
      .fit(normalizedFeatures)

    /*
      +--------+--------------------+--------------------+
      |gis_join|            features|         pcaFeatures|
      +--------+--------------------+--------------------+
      |G1200870|[0.54405509418675...|[0.80492832393268...|
      |G1200870|[0.55296738910269...|[0.80857487657638...|
      |G1200870|[0.55276483694551...|[0.76926659597088...|
      |G1200870|[0.55377759773141...|[0.79674422064382...|
      |G1200870|[0.54628316791573...|[0.71710709966826...|
      |G1200870|[0.55448653028154...|[0.77397137809210...|
      |G1200870|[0.55600567146040...|[0.68385808112508...|
      |G1200870|[0.55772736479643...|[0.66660908921373...|
      |G1200870|[0.55499291067449...|[0.75565165269005...|
      |G1200870|[0.55468908243872...|[0.72406202099241...|
      |G1200870|[0.55701843224630...|[0.70259295830020...|
      |G1200870|[0.55894267773951...|[0.65973792208690...|
      ...
      +--------+--------------------+--------------------+
     */
    val pc: DenseMatrix = pca.pc
    var pcaDF: Dataset[Row] = pca.transform(normalizedFeatures).select("gis_join", "features", "pcaFeatures");
    // val requiredNoOfPCs: Int = getNoPrincipalComponentsByVariance(pca, 0.95); // 6

    /* Disassembled PCA features column into individual columns.
      +--------+--------------------+--------------------+------------------+------------------+--------------------+--------------------+------------------+--------------------+
      |gis_join|            features|         pcaFeatures|     pcaFeatures_0|     pcaFeatures_1|       pcaFeatures_2|       pcaFeatures_3|     pcaFeatures_4|       pcaFeatures_5|
      +--------+--------------------+--------------------+------------------+------------------+--------------------+--------------------+------------------+--------------------+
      |G1200870|[0.54405509418675...|[0.80492832393268...| 0.804928323932684|1.2524834980466761|-0.19575432512666624|-0.03944625114810606|1.2090768601608095|0.040411909966895754|
      |G1200870|[0.55296738910269...|[0.80857487657638...|0.8085748765763834|1.2235243896765384|-0.21292887826797896|-0.02198527607780032|1.1302797473337691|0.008653301140702473|
      |G1200870|[0.55276483694551...|[0.76926659597087...|0.7692665959708778| 1.297008853789224|-0.14863430340801337|-0.04637046836915122|1.1779886744375148|  0.0545280473121077|
      |G1200870|[0.55377759773141...|[0.79674422064381...|0.7967442206438194|1.2373180484117852| -0.1858459251191183|-0.05321292641319979|1.1275219965290009| 0.04581155157437637|
      |G1200870|[0.54628316791573...|[0.71710709966825...|0.7171070996682597|1.3857646539960418|-0.04885903454878...|-0.11196356649642812| 1.239957766193588| 0.05632980370416987|
      |G1200870|[0.55448653028154...|[0.77397137809210...|0.7739713780921003|1.2681072652159826|-0.14264661986123855|-0.08646276453469218|1.1357663877134403| 0.05142487299813041|
      |G1200870|[0.55600567146040...|[0.68385808112507...|0.6838580811250755|1.4102913185089028| 0.02880129713901572|-0.20988835643907133| 1.220484723051305| 0.07826483591938016|
      ...
      |G1200860|[0.55134697184525...|[0.84536048028294...|0.8453604802829415|1.1649733636908406|  -0.251498842124316|-0.05398789321041379|1.1379273562164913|0.006627433190429924|
      |G1200860|[0.55033421105934...|[0.84497922527196...|0.8449792252719691|1.1720647495314305| -0.2684042852950131|-0.01816121020906...|1.1458699947581152|0.009816855921066535|
      |G1200860|[0.55235973263115...|[0.83180258697776...|0.8318025869777639|1.1888588297495093|  -0.247244637606236|-0.01792134094207...|1.1359619004162465| 0.00902030342487202|
      |G1200860|[0.55073931537370...|[0.84546368981010...|0.8454636898101023|1.1706852781462835| -0.2708025633589945|-0.00719615226018...|1.1392678220552446|0.008471343752829472|
      |G1200860|[0.54911889811626...|[0.84514470930467...|0.8451447093046771|1.1706297920180826| -0.2708819367777809|-0.00756457734791...|1.1372865563117458|0.014087994825427497|
      ...
      +--------+--------------------+--------------------+------------------+------------------+--------------------+--------------------+------------------+--------------------+
    */
    val disassembler = new VectorDisassembler().setInputCol("pcaFeatures")
    pcaDF = disassembler.transform(pcaDF)
    pcaDF
  }

  /**
   * Determines the minimum number of principal components needed to capture a target variance between observations.
   * @param pca The Spark PCA model that has been fit to the data.
   * @param targetVariance The target variance [0, 1.0] (suggested: 0.95) we wish to capture.
   * @return The number (K) of principle components which capture, for example, 95% of the variance between observations.
   */
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

}
