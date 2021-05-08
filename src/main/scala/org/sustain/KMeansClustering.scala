package org.sustain

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, collect_list, row_number}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

import java.io.{File, PrintWriter}
import java.util

class KMeansClustering {

  def runClustering(spark: SparkSession, inputCollection: Dataset[Row], clusteringFeatures: Array[String], clusteringK: Int): Unit = {

    import spark.implicits._

    /* We start off with this Dataframe from the PCA output:
      +--------+-------------------+------------------+-------------------+--------------------+------------------+-------------------+
      |gis_join|           avg_pc_0|          avg_pc_1|           avg_pc_2|            avg_pc_3|          avg_pc_4|           avg_pc_5|
      +--------+-------------------+------------------+-------------------+--------------------+------------------+-------------------+
      |G1701390| 0.4158020936974724|1.1287812290352837|0.08484897415159401|-0.36232560204050596| 0.971158581704508| 0.3292011032114153|
      |G4200570| 0.3104423560186878|1.1214304733142375|0.16773854547936934| -0.3160498945338415|0.9289753853324734|  0.364798093601345|
      |G1801810| 0.4281583709651544|1.1269770094714264|0.09164693786985073| -0.3585017356023207|0.9595259561164894|0.33325110624398663|
      |G1701230| 0.3940285496472165|1.0964354704796004| 0.0730388748867706| -0.3860192369348192|0.9602560852897506|0.32299471292026727|
      |G3100050| 0.2946745168790351|0.8888264467749074| 0.3575415331802127|-0.24547688421297456|0.9640098330128868|0.31425285813388415|
      |G1901890|0.44782911725726976|0.9984533406366115|0.10817368191999042| -0.4163631071256441|0.9119211591625958|0.30074001508335846|
      |G3600770| 0.5576374023453231|1.0073503484420696|0.12931249839589687| -0.3116525232658502|0.9244468108363925| 0.3531358094469876|
      |G4600710|0.39261240555802535|0.9216624497708813|0.22624592034855084| -0.3509425272917775|0.9832587487915269|0.30294115994607734|
      |G2700790| 0.4432941927736439|1.0032628666592123| 0.0805827963993705|   -0.41699363311637|0.8980082948681012| 0.2958308086472813|
      |G3800590| 0.5290105425278709|0.9133360843622881|0.17210550926633586|-0.41950735729077504|0.9287485099489983|0.27868281242930765|
      ...
      +--------+-------------------+------------------+-------------------+--------------------+------------------+-------------------+
     */
    var clusteringCollection = inputCollection

    /* Assemble pc features into single column of type vector:
      +--------+--------------------+
      |gis_join|            features|
      +--------+--------------------+
      |G1201050|[0.06377826865377...|
      |G4804550|[0.07013847183673...|
      |G4500890|[0.06205512757281...|
      |G4701690|[0.26897356018054...|
      |G1700030|[0.25943378119348...|
      |G2001030|[0.40963069132444...|
      |G1701390|[0.41580209369747...|
      |G4200570|[0.31044235601868...|
      |G1801810|[0.42815837096515...|
      ...
      +--------+--------------------+
     */
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(clusteringFeatures)
      .setOutputCol("features")
    val withFeaturesAssembled: Dataset[Row] = assembler.transform(clusteringCollection).select("gis_join", "features")

    /* KMeans clustering centers: K = sqrt(N = 3192) = 56 centers
      [0.2516285197122672,1.2077966459300173,0.10184393694730279,-0.2832774504883632,0.9575598806493245,0.3474121497943072]
      [0.3846041603005205,0.5111460686812306,0.6562840345366265,-0.12649469690747978,0.9999756803184017,0.29672545856505655]
      [0.3473555262419151,0.9228535371047386,0.32849860847099754,-0.24316970595244403,0.9815467689749992,0.3149804772098435]
      [0.08073848277123541,1.3228816046847527,0.04054138630382662,-0.08089940823098363,1.0850326683632194,0.1931426866644562]
      [0.5292323019888051,0.955174793334104,0.12370416020077485,-0.39634298547682106,0.914124072758712,0.2868526450082559]
      ...
      [0.02514500239201011,0.9809745259146007,0.411014453894612,0.009504899351576961,0.9185787192018573,0.31316348779096365]
     */
    val kMeans: KMeans = new KMeans().setK(clusteringK).setSeed(1L)
    val kMeansModel: KMeansModel = kMeans.fit(withFeaturesAssembled)
    val centers: Array[Vector] = kMeansModel.clusterCenters
    println(">>> Cluster centers:\n")
    centers.foreach { println }

    /* Get cluster predictions
      +--------+--------------------+----------+
      |gis_join|            features|prediction|
      +--------+--------------------+----------+
      |G4500890|[0.06205512757281...|         7|
      |G4701690|[0.26897356018054...|         0|
      |G1700030|[0.25943378119348...|         0|
      |G2001030|[0.40963069132444...|        18|
      |G1701390|[0.41580209369747...|        18|
      |G4200570|[0.31044235601868...|        15|
      |G1801810|[0.42815837096515...|        18|
      |G1701230|[0.39402854964721...|        18|
      |G3100050|[0.29467451687903...|        47|
      |G1901890|[0.44782911725726...|        20|
      ...
      +--------+--------------------+----------+
     */
    val predictions: Dataset[Row] = kMeansModel.transform(withFeaturesAssembled)
      .select("gis_join", "features", "prediction")

    /* Calculate distances to cluster center
      +--------+----------+--------------------+
      |gis_join|prediction|            distance|
      +--------+----------+--------------------+
      |G4804230|         2|5.838333109652426E-4|
      |G5600390|         1|7.204866911420771E-4|
      |G1701150|         1|0.001248523509027...|
      |G0601030|         2|0.002464164336879857|
      |G3701230|         2|1.975126007273414...|
      |G3700690|         2|1.835193565265453E-4|
      |G3701070|         2|2.455512153503290...|
      |G4803630|         0|0.003182987447111...|
      |G5108200|         0|3.312913423429138E-4|
      ...
      +--------+----------+--------------------+
     */
    var distances: Dataset[Row] = predictions.map( row => {
      val prediction:   Int    = row.getInt(2)        // Cluster prediction
      val featuresVect: Vector = row.getAs[Vector](1) // Normalized features
      val centersVect:  Vector = centers(prediction)  // Normalized cluster centers
      val distance = Vectors.sqdist(featuresVect, centersVect) // Squared dist between features and cluster centers

      (row.getString(0), row.getInt(2), distance) // (String, Int, Double)
    }).toDF("gis_join", "prediction", "distance").as("distances")

    /* Partition by prediction, find the minimum distance value, and pair back with original dataframe.
      +--------+----------+--------------------+
      |gis_join|prediction|            distance|
      +--------+----------+--------------------+
      |G0800790|        31|5.262008880718873E-4|
      |G1900610|        53|1.901067647533151E-4|
      |G4701850|        34|3.589381052639582...|
      |G3601230|        28|4.974901361715556E-4|
      |G3100270|        26|2.594371668810202E-4|
      |G0600970|        27|0.001172388772051...|
      |G3000810|        44|9.366204912018388E-4|
      |G1300830|        12|3.387004285991619E-4|
      |G4804490|        22|1.501039078631702...|
      |G3000030|        47|6.315135696048954E-4|
      ...
      +--------+----------+--------------------+
     */
    val closestPoints = Window.partitionBy("prediction").orderBy(col("distance").asc)
    distances = distances.withColumn("row",row_number.over(closestPoints))
      .where($"row" === 1).drop("row")
    distances.show()

    val clusterRows: Dataset[Row] = predictions.select("gis_join", "prediction")
      .groupBy(col("prediction"))
      .agg(collect_list("gis_join"))

    /* Collect into Array[(<gis_join>, <cluster_id>)]
      [
        (G0800790,31)
        (G1900610,53)
        ...
        (G4900230,36)
      ]
     */
    val gisJoinCenters: Array[(String, Int)] = distances.collect().map(
      row => (row.getString(0), row.getInt(1))
    )

    /* Collect into Array[(<cluster_id>, <list_gis_joins>)]
      [
        (0, [G0800790, G0800793, ..., G0800798],
        (1, [G0804591, G0845919, ..., G0845978],
        ...
        (55, [G0854592, G0855913, ..., G0855972]
      ]
     */
    val clusters: Array[(Int, util.List[String])] = clusterRows.collect().map(row => {
      (row.getInt(0), row.getList[String](1))
    })

    writeClustersToFile(gisJoinCenters, clusters)
  }

  /**
   * Write a CSV file containing all the clusters computed from PCA, along with their cluster centers in the format:
   *  cluster_id,gis_join,is_center
   * @param centers The list of cluster center GISJoins
   * @param clusters The list of cluster GISJoins
   */
  def writeClustersToFile(centers: Array[(String, Int)], clusters: Array[(Int, util.List[String])]): Unit = {
    val pw: PrintWriter = new PrintWriter(new File("/s/parsons/b/others/sustain/caleb/transfer-learning/clusters.csv"))
    pw.write("cluster_id,gis_join,is_center\n")
    for (cluster <- clusters) {
      val clusterId: Int = cluster._1
      cluster._2.forEach(
        gisJoin => {
          pw.write("%d,%s,%s\n".format(clusterId, gisJoin, isCenterGisJoin(gisJoin, centers)))
        }
      )
    }
    pw.close()
  }

  /**
   * Determines if a GISJoin is the cluster's centroid.
   * @param gisJoin The candidate in question
   * @param centers the list of cluster centroids
   * @return String representation of boolean to be printed to file
   */
  def isCenterGisJoin(gisJoin: String, centers: Array[(String, Int)]): String = {
    for (center <- centers) {
      if (center._1 == gisJoin) {
        return "true"
      }
    }
    "false"
  }

}
