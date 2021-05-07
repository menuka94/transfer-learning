package org.sustain

import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset, Row, RowFactory}
import org.apache.spark.sql.functions.{col, collect_list, min, row_number, struct}
import org.apache.spark.ml.regression.LinearRegression
import com.mongodb.spark.MongoSpark
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Dataset, Row, SparkSession}

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

@SerialVersionUID(114L)
class Experiment() extends Serializable {

  def transferLearning(sparkMaster: String, appName: String, mongosRouters: Array[String], mongoPort: String,
                       database: String, collection: String, clusteringFeatures: Array[String], clusteringYMDH: Long,
                       clusteringTimestep: Long, clusteringK: Int, regressionFeatures: Array[String],
                       regressionLabel: String, pcaFeatures: Array[String]): Unit = {

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

    val pca: PrincipleComponentAnalysis = new PrincipleComponentAnalysis()

    /* Run PCA on the collection and get the DF containing the principle components
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
    val pcaDF: Dataset[Row] = pca.runPCA(sparkConnector, pcaFeatures)

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
    var mongoCollection: Dataset[Row] = MongoSpark.load(sparkConnector)
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

    // Create LR models for cluster centroid GISJoins
    val centroidModels: Array[CentroidModel] = new Array[CentroidModel](gisJoinCenters.length)
    for (i <- gisJoinCenters.indices) {
      val gisJoin: String = gisJoinCenters(i)._1
      val clusterId: Int = gisJoinCenters(i)._2
      val mongoHost: String = mongosRouters(clusterId % mongosRouters.length) // choose a mongos router
      val centroidModel: CentroidModel = new CentroidModel(sparkMaster, mongoHost, mongoPort,
        database, collection, regressionLabel, regressionFeatures, gisJoin, clusterId)
      centroidModels(clusterId) = centroidModel
    }

    try {
      // Kick off training of LR models for center GISJoins
      centroidModels.foreach(model => model.start())

      // Wait until models are done being trained
      centroidModels.foreach(model => model.join())
    } catch {
      case e: java.lang.IllegalMonitorStateException => println("\n\nn>>>Caught IllegalMonitorStateException!")
    }

    println("\n\n>>> Initial center models done training\n")

    // Sort trained models by their predicted cluster ID
    scala.util.Sorting.quickSort(centroidModels)

    /* Group together GISJoins which belong to the same cluster
      +----------+----------------------+
      |prediction|collect_list(gis_join)|
      +----------+----------------------+
      |        31|  [G3800670, G38009...|
      |        53|  [G5500510, G35000...|
      |        34|  [G3100250, G31013...|
      |        28|  [G1701150, G36000...|
      |        26|  [G1301790, G13022...|
      |        27|  [G4100390, G13000...|
      |        44|  [G5500150, G55008...|
      |        12|  [G7200540, G72001...|
      |        22|  [G3101030, G31014...|
      |        47|  [G4800890, G48046...|
      |         1|  [G1700990, G55012...|
      ...
      +----------+----------------------+
     */
    val clusterRows: Dataset[Row] = predictions.select("gis_join", "prediction")
      .groupBy(col("prediction"))
      .agg(collect_list("gis_join"))

    // Collect clustered GISJoins into memory
    val clusters: Array[(Int, Array[String])] = clusterRows.collect()
      .map(
        row => {
          val gisJoins: Array[String] = row.getList[String](1).asScala.toArray
          (row.getInt(0), gisJoins)
        }
      )

    // Build K queues of models to be trained, 1 queue per cluster
    // For 3192 GISJoins this is sqrt(3192) = ~56 queues, each with ~57 models to be trained (since cluster sizes vary,
    // some queues may be shorter and others larger)
    val clustersQueues: Array[ClusterLRModels] = new Array[ClusterLRModels](gisJoinCenters.length)
    clusters.foreach(
      cluster => { // (31, [G3800670, G38009, ...])
        val clusterId: Int = cluster._1
        val gisJoins: Array[String] = cluster._2
        val center: (String, Int) = gisJoinCenters(clusterId)
        val centerGisJoin: String = center._1
        val trainedRegression: CentroidModel = centroidModels(clusterId)
        val trainedModel: LinearRegression = trainedRegression.linearRegression
        val mongoRouterHost: String = mongosRouters(clusterId % mongosRouters.length)
        clustersQueues(clusterId) = new ClusterLRModels(sparkMaster, mongoRouterHost, mongoPort, database, collection,
          clusterId, gisJoins, trainedModel, centerGisJoin, regressionFeatures, regressionLabel, mongoCollection)
      }
    )
    /*
        for (i <- gisJoinCenters.indices) {
          val center: (String, Int) = gisJoinCenters(i)
          val centerGisJoin: String = center._1
          val clusterId: Int = center._2
          val trainedRegression: CentroidModel = centroidModels(clusterId)
          val trainedModel: LinearRegression = trainedRegression.linearRegression
          val mongoRouterHost: String = mongosRouters(clusterId % mongosRouters.length)

          // Create a new Queue
          val gisJoinList: ListBuffer[String] = new ListBuffer[String]()

          // Get only gisJoins for this clusterId and that are not the center gisJoin, and create regression models from
          // the trained centroid model, adding to the model queue

          predictions.select("gis_join", "prediction")
            .filter(col("prediction") === clusterId && col("gis_join") =!= centerGisJoin)
            .collect()
            .foreach(row => { // Iterates over cluster_size gisJoins
              gisJoinList += row.getString(0)
            })

          val clusterModels: ClusterLRModels = new ClusterLRModels(sparkMaster, mongoRouterHost, mongoPort, database,
            collection, clusterId, gisJoinList.toArray, trainedModel, regressionFeatures, regressionLabel, mongoCollection)
          clustersQueues(i) = clusterModels
    }
    */


    // --- DEBUGGING ---
    println("\n\n>>> MODEL QUEUES <<<\n")
    clustersQueues.foreach{ println }


    try {
      clustersQueues.foreach(queue => queue.start())
      clustersQueues.foreach(queue => queue.wait())
    } catch {
      case e: java.lang.IllegalMonitorStateException => println("\n\nn>>>Caught IllegalMonitorStateException!")
    }



    /*
    // Create new Regression model and initialize it with the already-trained model
    val testGisJoin: String = "G0601030"
    val testClusterId: Int = 2
    val newRegressionModel: Regression = new Regression(testGisJoin, testClusterId)
    val trainedRegression: Regression = regressionModels(testClusterId)
    newRegressionModel.linearRegression = trainedRegression.linearRegression.copy(new ParamMap())

    println("\n\n>>> New Regression Model's ParamMap: " + newRegressionModel.linearRegression.extractParamMap().toString())
    */
    /*
    // Sort trained models by their predicted cluster ID


    val allGisJoins: Array[(String, Int)] = predictions.map(row => {
      (row.getString(0), row.getInt(2))
    }).collect()

    val allRegressionModels: Array[Regression] = new Array[Regression](allGisJoins.length)
    for (i <- allRegressionModels.indices) {
      val gisJoin: String = allGisJoins(i)._1
      val clusterId: Int = allGisJoins(i)._2
      val regression: Regression = new Regression(gisJoin, clusterId)
      regression.linearRegression =
      allRegressionModels(i) = regression
    }

    println("\n\n>>> Training remaining " + allRegressionModels.length + " models")

    try {
      // Kick off training of LR models for center GISJoins
      for (i <- allRegressionModels.indices) {
        allRegressionModels(i).start()
      }

      // Wait until models are done being trained
      for (i <- allRegressionModels.indices) {
        allRegressionModels(i).join()
      }

    } catch {
      case e: java.lang.IllegalMonitorStateException => println("\n\nn>>>Caught IllegalMonitorStateException!")
    }

     */


  }

}
