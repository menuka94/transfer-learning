package org.sustain

import com.mongodb.spark.MongoSpark
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel, VectorAssembler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, row_number}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

class KMeansClustering {

  def runClustering(sparkConnector: SparkSession, inputCollection: Dataset[Row], clusteringFeatures: Array[String], clusteringYMDH: Long,
                    clusteringTimestep: Long, clusteringK: Int): Array[(String, Int)] = {

    import sparkConnector.implicits._

    /* Read collection into a DataSet[Row], dropping null rows
          +--------+-------------------+--------+-------------------------+
          |gis_join|year_month_day_hour|timestep|temp_surface_level_kelvin|
          +--------+-------------------+--------+-------------------------+
          |G4804230|         2010010100|       0|        281.4640808105469|
          |G5600390|         2010010100|       0|        265.2140808105469|
          |G1701150|         2010010100|       0|        265.7140808105469|
          |G0601030|         2010010100|       0|        282.9640808105469|
          |G3701230|         2010010100|       0|        279.2140808105469|
          |G3700690|         2010010100|       0|        280.8390808105469|
          |G3701070|         2010010100|       0|        280.9640808105469|
          |G4803630|         2010010100|       0|        275.7140808105469|
          |G5108200|         2010010100|       0|        273.4640808105469|
          |G4801170|         2010010100|       0|        269.3390808105469|
          +--------+-------------------+--------+-------------------------+
         */
    var mongoCollection = inputCollection
    mongoCollection = mongoCollection.select("gis_join", "year_month_day_hour", "timestep", "temp_surface_level_kelvin")
      .na.drop()

    /* Take only 1 entry per GISJOIN across all timesteps for clustering
      +--------+-------------------------+
      |gis_join|temp_surface_level_kelvin|
      +--------+-------------------------+
      |G4804230|        281.4640808105469|
      |G5600390|        265.2140808105469|
      |G1701150|        265.7140808105469|
      |G0601030|        282.9640808105469|
      |G3701230|        279.2140808105469|
      |G3700690|        280.8390808105469|
      |G3701070|        280.9640808105469|
      |G4803630|        275.7140808105469|
      |G5108200|        273.4640808105469|
      |G4801170|        269.3390808105469|
      +--------+-------------------------+
     */
    val clusteringCollection: Dataset[Row] = mongoCollection.filter(
      col("year_month_day_hour") === clusteringYMDH && col("timestep") === clusteringTimestep
    ).select("gis_join", "temp_surface_level_kelvin")

    /* Assemble features into single column
      +--------+-------------------------+-------------------+
      |gis_join|temp_surface_level_kelvin|           features|
      +--------+-------------------------+-------------------+
      |G4804230|        281.4640808105469|[281.4640808105469]|
      |G5600390|        265.2140808105469|[265.2140808105469]|
      |G1701150|        265.7140808105469|[265.7140808105469]|
      |G0601030|        282.9640808105469|[282.9640808105469]|
      |G3701230|        279.2140808105469|[279.2140808105469]|
      |G3700690|        280.8390808105469|[280.8390808105469]|
      |G3701070|        280.9640808105469|[280.9640808105469]|
      |G4803630|        275.7140808105469|[275.7140808105469]|
      |G5108200|        273.4640808105469|[273.4640808105469]|
      |G4801170|        269.3390808105469|[269.3390808105469]|
      +--------+-------------------------+-------------------+
     */
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(clusteringFeatures)
      .setOutputCol("features")
    val withFeaturesAssembled: Dataset[Row] = assembler.transform(clusteringCollection)

    /* Normalize features
      +--------+--------------------+
      |gis_join|            features|
      +--------+--------------------+
      |G4804230|[0.6709129511677282]|
      |G5600390|[0.3949044585987261]|
      |G1701150|[0.4033970276008492]|
      |G0601030|[0.6963906581740976]|
      |G3701230|[0.6326963906581741]|
      |G3700690|[0.6602972399150743]|
      |G3701070|[0.6624203821656051]|
      |G4803630|[0.5732484076433121]|
      |G5108200| [0.535031847133758]|
      |G4801170|[0.46496815286624...|
      +--------+--------------------+
     */
    val minMaxScaler: MinMaxScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("normalized_features")
    val minMaxScalerModel: MinMaxScalerModel = minMaxScaler.fit(withFeaturesAssembled)
    var normalizedFeatures: Dataset[Row] = minMaxScalerModel.transform(withFeaturesAssembled)
    normalizedFeatures = normalizedFeatures.drop("features")
    normalizedFeatures = normalizedFeatures.withColumnRenamed("normalized_features", "features")
      .select("gis_join", "features")
    println(">>> With normalized features:\n")


    /* KMeans clustering centers
      [0.5168304366844847]
      [0.3680625754467921]
      [0.6467503082873386]
      [0.21075872847369662]
      [0.8369497523000703]
     */
    val kMeans: KMeans = new KMeans().setK(clusteringK).setSeed(1L)
    val kMeansModel: KMeansModel = kMeans.fit(normalizedFeatures)
    val centers: Array[Vector] = kMeansModel.clusterCenters
    println(">>> Cluster centers:\n")
    centers.foreach { println }


    /* Get cluster predictions
      +--------+--------------------+----------+
      |gis_join|            features|prediction|
      +--------+--------------------+----------+
      |G4804230|[0.6709129511677282]|         2|
      |G5600390|[0.3949044585987261]|         1|
      |G1701150|[0.4033970276008492]|         1|
      |G0601030|[0.6963906581740976]|         2|
      |G3701230|[0.6326963906581741]|         2|
      |G3700690|[0.6602972399150743]|         2|
      |G3701070|[0.6624203821656051]|         2|
      |G4803630|[0.5732484076433121]|         0|
      |G5108200| [0.535031847133758]|         0|
      |G4801170|[0.46496815286624...|         0|
      +--------+--------------------+----------+
     */
    val predictions: Dataset[Row] = kMeansModel.transform(normalizedFeatures)
    println(">>> Predictions centers:\n")

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
      |G2001630|         1|5.760295484876104E-7|
      |G4601270|         3|3.222216079740129...|
      |G1201050|         4|1.863697172494980...|
      |G2800330|         2|6.529902553778138E-7|
      |G3900550|         0| 8.22412844134513E-7|
      +--------+----------+--------------------+
     */
    val closestPoints = Window.partitionBy("prediction").orderBy(col("distance").asc)
    distances = distances.withColumn("row",row_number.over(closestPoints))
      .where($"row" === 1).drop("row")
    distances.show()

    /* Collect into Array[(<gis_join>, <prediction>)]
      [
        ("G2001630", 1),
        ("G4601270", 3),
        ("G1201050", 4),
        ("G2800330", 2),
        ("G3900550", 0),
      ]
     */
    val gisJoinCenters: Array[(String, Int)] = distances.collect().map(
      row => (row.getString(0), row.getInt(1))
    )

    gisJoinCenters

  }

}
