package org.sustain

import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions.{avg, col}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import java.io.{BufferedWriter, File, FileWriter}
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

@SerialVersionUID(114L)
class Experiment() extends Serializable {

  def transferLearning(sparkMaster: String, appName: String, mongosRouters: Array[String], mongoPort: String,
                       database: String, collection: String, regressionFeatures: Array[String],
                       regressionLabel: String, profileOutput: String, centroidStatsCSV: String,
                       clusterModelStatsCSV: String, pcaClusters: Array[PCACluster]): Unit = {

    val profiler: Profiler = new Profiler()
    val experimentTaskId: Int = profiler.addTask("Experiment")
    scala.util.Sorting.quickSort(pcaClusters)

    val conf: SparkConf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName(appName)
      .set("spark.executor.cores", "8")
      .set("spark.executor.memory", "16G")
      .set("spark.mongodb.input.uri", "mongodb://%s:%s/".format(mongosRouters(0), mongoPort)) // default mongos router
      .set("spark.mongodb.input.database", database) // sustaindb
      .set("spark.mongodb.input.collection", collection) // noaa_nam
      //.set("mongodb.keep_alive_ms", "100000") // Important! Default is 5000ms, and stream will prematurely close

    // Create the SparkSession and ReadConfig
    val sparkSession: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()

    import sparkSession.implicits._ // For the $()-referenced columns

    // Create LR models for cluster centroid GISJoins
    val centroidModels: Array[CentroidModel] = new Array[CentroidModel](pcaClusters.length)
    for (cluster: PCACluster <- pcaClusters) {
      val mongoHost: String = mongosRouters(cluster.clusterId % mongosRouters.length) // choose a mongos router
      val mongoUri: String = "mongodb://%s:%s/".format(mongoHost, mongoPort)
      centroidModels(cluster.clusterId) = new CentroidModel(sparkMaster, mongoUri, database, collection,
        regressionLabel, regressionFeatures, cluster.centerGisJoin, cluster.clusterId, sparkSession, profiler, centroidStatsCSV)
    }

    writeCentroidStatsCSVHeader(centroidStatsCSV)

    try {
      // Kick off training of LR models for center GISJoins
      centroidModels.foreach(model => model.start())

      // Wait until models are done being trained
      centroidModels.foreach(model => model.join())
    } catch {
      case e: java.lang.IllegalMonitorStateException => println("\n\n>>>Caught IllegalMonitorStateException!")
    }

    println("\n\n>>> Initial center models done training\n")

    println("\n\n>>> Beginning to train cluster models\n")

    // Sort trained models by their predicted cluster ID
    scala.util.Sorting.quickSort(centroidModels)
    writeClusterModelsStatsCSVHeader(clusterModelStatsCSV)

    // Create ClusterLRModels models for cluster GISJoins
    val clusterModels: Array[ClusterLRModels] = new Array[ClusterLRModels](pcaClusters.length)
    for (cluster: PCACluster <- pcaClusters) {
      val mongoHost: String = mongosRouters(cluster.clusterId % mongosRouters.length) // choose a mongos router
      val mongoUri: String = "mongodb://%s:%s/".format(mongoHost, mongoPort)
      val centroidModel: CentroidModel = centroidModels(cluster.clusterId)

      clusterModels(cluster.clusterId) = new ClusterLRModels(sparkMaster, mongoUri, database, collection, cluster.clusterId,
        cluster.clusterGisJoins.toArray, centroidModel.linearRegression, cluster.centerGisJoin, regressionFeatures,
        regressionLabel, profiler, sparkSession, clusterModelStatsCSV)
    }

    try {
      // Kick off training of LR models for all clusters
      clusterModels.foreach(cluster => cluster.start())

      // Wait until models are done being trained
      clusterModels.foreach(cluster => cluster.join())
    } catch {
      case e: java.lang.IllegalMonitorStateException =>
        println("\n\n>>>Caught IllegalMonitorStateException!")
        println(e.getMessage)

        // Safe cleanup
        sparkSession.close()
        return

    }

    profiler.finishTask(experimentTaskId)
    profiler.writeToFile(profileOutput)
    profiler.close()
  }

  /**
   * Builds all ~3000 models sequentially without transfer learning, taking profiling stats for each model built.
   */
  def sequentialTraining(sparkMaster: String, appName: String, mongosRouters: Array[String], mongoPort: String,
                         database: String, collection: String, regressionFeatures: Array[String],
                         regressionLabel: String, sequentialStatsCSV: String, profileOutput: String, gisJoins: Array[String]): Unit = {


    val profiler: Profiler = new Profiler()
    val experimentTaskId: Int = profiler.addTask("Experiment")

    val conf: SparkConf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName(appName)
      .set("spark.executor.cores", "6")
      .set("spark.executor.memory", "32G")
      .set("spark.mongodb.input.uri", "mongodb://%s:%s/".format(mongosRouters(0), mongoPort)) // default mongos router
      .set("spark.mongodb.input.database", database) // sustaindb
      .set("spark.mongodb.input.collection", collection) // noaa_nam

    // Create the SparkSession and ReadConfig
    val sparkSession: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()

    writeSequentialModelHeader(sequentialStatsCSV)

    val clusterModels: Array[ClusterLRModels] = new Array[ClusterLRModels](56)
    val clusters: Array[ArrayBuffer[String]] = loadDefaultClusters(gisJoins)

    for (i <- clusters.indices) {
      val mongoHost: String = mongosRouters(i % mongosRouters.length) // choose a mongos router for cluster
      val cluster: Array[String] = clusters(i).toArray
      val mongoUri: String = "mongodb://%s:%s/".format(mongoHost, mongoPort)
      val lr: LinearRegression = new LinearRegression()
        .setFitIntercept(true)
        .setMaxIter(10)
        .setLoss("squaredError")
        .setSolver("l-bfgs")
        .setStandardization(true)

      clusterModels(i) = new ClusterLRModels(sparkMaster, mongoUri, database, collection, i, cluster, lr, cluster(0),
        regressionFeatures, regressionLabel, profiler, sparkSession, sequentialStatsCSV)
    }

    try {
      // Kick off training of LR models for all clusters
      clusterModels.foreach(cluster => cluster.start())

      // Wait until models are done being trained
      clusterModels.foreach(cluster => cluster.join())
    } catch {
      case e: java.lang.IllegalMonitorStateException =>
        println("\n\n>>>Caught IllegalMonitorStateException!")
        println(e.getMessage)

        // Safe cleanup
        sparkSession.close()
        return

    }


/*

    // Train all models
    var modelsTrained: Int = 0
    gisJoins.foreach(
      gisJoin => {

        // >>> Begin Task to persist the collection
        val persistTaskName: String = "Load Dataframe + Select + Filter + Vector Assemble + Persist + Count;gisJoin=%s".format(gisJoin)
        val persistTaskId: Int = profiler.addTask(persistTaskName)

        var mongoCollection: Dataset[Row] = MongoSpark.load(sparkSession, readConfig).select(
          "gis_join", "relative_humidity_percent", "timestep", "temp_surface_level_kelvin"
        ).na.drop().filter(
          col("gis_join") === gisJoin && col("timestep") === 0
        ).withColumnRenamed(regressionLabel, "label")

        val assembler: VectorAssembler = new VectorAssembler()
          .setInputCols(regressionFeatures)
          .setOutputCol("features")
        mongoCollection = assembler.transform(mongoCollection).persist()
        val numRecords: Long = mongoCollection.count()

        // <<< End Task to persist the collection
        profiler.finishTask(persistTaskId)

        mongoCollection = mongoCollection.filter(col("gis_join") === gisJoin)

        // Split input into testing set and training set:
        // 80% training, 20% testing, with random seed of 42
        val Array(train, test): Array[Dataset[Row]] = mongoCollection.randomSplit(Array(0.8, 0.2), 42)

        // Create Linear Regression Estimator
        val linearRegression: LinearRegression = new LinearRegression()


        // >>> Begin Task to fit training set
        val fitTaskName: String = "Fit Training Set"
        val fitTaskId: Int = profiler.addTask(fitTaskName)

        // Train LR model
        val begin: Long = System.currentTimeMillis()
        val lrModel: LinearRegressionModel = linearRegression.fit(train)
        val end: Long = System.currentTimeMillis()
        val iterations: Int = lrModel.summary.totalIterations

        // <<< End Task to fit training set
        profiler.finishTask(fitTaskId, System.currentTimeMillis())

        // Establish a Regression Evaluator for RMSE
        val evaluator: RegressionEvaluator = new RegressionEvaluator()
          .setMetricName("rmse")
        val predictions: DataFrame = lrModel.transform(test)
        val rmse: Double = evaluator.evaluate(predictions)

        println("\n\n>>> Test set RMSE for GISJoin [%d]: %s: %.4f\n".format(modelsTrained+1, gisJoin, rmse))

        writeSequentialModelStats(sequentialStatsCSV, gisJoin, end-begin, rmse, iterations, numRecords)
        mongoCollection.unpersist()
        modelsTrained += 1
      }
    )*/

    // <<< End Task for Experiment
    profiler.finishTask(experimentTaskId)
    profiler.writeToFile(profileOutput)
    profiler.close()
  }

  def loadDefaultClusters(gisJoins: Array[String]): Array[ArrayBuffer[String]] = {
    val clusters: Array[ArrayBuffer[String]] = new Array[ArrayBuffer[String]](56)
    for (i <- clusters.indices) {
      clusters(i) = new ArrayBuffer[String]()
    }
    for (i <- gisJoins.indices) {
      val clusterIndex: Int = i % 56
      clusters(clusterIndex) += gisJoins(i)
    }

    for (i <- clusters.indices) {
      println("Cluster Size:", clusters(i).size)
    }
    clusters
  }

  def loadGisJoins(filename: String): Array[String] = {
    val gisJoins: ArrayBuffer[String] = new ArrayBuffer[String]()

    val bufferedSource = Source.fromFile(filename)
    var lineNumber: Int = 0
    for (line <- bufferedSource.getLines) {
      if (lineNumber > 0) {
        val cols = line.split(",").map(_.trim)
        val gisJoin: String = cols(1)
        gisJoins += gisJoin
      }
      lineNumber += 1
    }
    bufferedSource.close
    gisJoins.toArray
  }

  def writeCentroidStatsCSVHeader(filename: String): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File(filename),
      )
    )
    bw.write("gis_join,cluster_id,number_records,time_ms,rmse,iterations,best_reg_param,best_tolerance,best_epsilon\n")
    bw.close()
  }

  def writeClusterModelsStatsCSVHeader(filename: String): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File(filename),
      )
    )
    bw.write("gis_join,cluster_id,time_ms,iterations,rmse\n")
    bw.close()
  }

  def loadClusters(filename: String, numClusters: Int): Array[PCACluster] = {
    val pcaClusters: Array[PCACluster] = new Array[PCACluster](numClusters)

    val bufferedSource = Source.fromFile(filename)
    var lineNumber: Int = 0
    for (line <- bufferedSource.getLines) {
      if (lineNumber > 0) {
        val cols = line.split(",").map(_.trim)
        val clusterId: Int = cols(0).toInt
        val gisJoin: String = cols(1)
        val isCenter: Boolean = cols(2).toBoolean
        if (pcaClusters(clusterId) == null) {
          pcaClusters(clusterId) = new PCACluster()
        }

        if (pcaClusters(clusterId).clusterId == -1) {
          pcaClusters(clusterId).clusterId = clusterId
        }

        if (isCenter) {
          pcaClusters(clusterId).centerGisJoin = gisJoin
        } else {
          pcaClusters(clusterId).clusterGisJoins += gisJoin
        }
      }
      lineNumber += 1
    }
    bufferedSource.close

    pcaClusters
  }

  def pcaClustering(sparkMaster: String, appName: String, mongosRouters: Array[String], mongoPort: String,
                    database: String, collection: String, clusteringFeatures: Array[String], clusteringK: Int,
                    pcaFeatures: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName(appName)
      .set("spark.executor.cores", "8")
      .set("spark.executor.memory", "16G")
      .set("spark.mongodb.input.uri", "mongodb://%s:%s/".format(mongosRouters(0), mongoPort))
      .set("spark.mongodb.input.database", database)
      .set("spark.mongodb.input.collection", collection)
      .set("mongodb.keep_alive_ms", "100000") // Important! Default is 5000ms, and stream will prematurely close

    // Create the SparkSession and ReadConfig
    val sparkConnector: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate() // For the $()-referenced columns

    import sparkConnector.implicits._

    /* Run PCA on the collection and get the DF containing the principle components
      +--------+--------------------+--------------------+------------------+------------------+--------------------+--------------------+------------------+--------------------+
      |gis_join|            features|         pcaFeatures|     pcaFeatures_0|     pcaFeatures_1|       pcaFeatures_2|       pcaFeatures_3|     pcaFeatures_4|       pcaFeatures_5|
      +--------+--------------------+--------------------+------------------+------------------+--------------------+--------------------+------------------+--------------------+
      |G1200870|[0.54405509418675...|[0.80492832393268...| 0.804928323932684|1.2524834980466761|-0.19575432512666624|-0.03944625114810606|1.2090768601608095|0.040411909966895754|
      |G1200870|[0.55296738910269...|[0.80857487657638...|0.8085748765763834|1.2235243896765384|-0.21292887826797896|-0.02198527607780032|1.1302797473337691|0.008653301140702473|
      |G1200870|[0.55276483694551...|[0.76926659597087...|0.7692665959708778| 1.297008853789224|-0.14863430340801337|-0.04637046836915122|1.1779886744375148|  0.0545280473121077|
      |G1200870|[0.55377759773141...|[0.79674422064381...|0.7967442206438194|1.2373180484117852| -0.1858459251191183|-0.05321292641319979|1.1275219965290009| 0.04581155157437637|
      |G1200870|[0.54628316791573...|[0.71710709966825...|0.7171070996682597|1.3857646539960418|-0.04885903454878...|-0.11196356649642812| 1.239957766193588| 0.05632980370416987|
      |G1200870|[0.55448653028154...|[0.77397137809210...|0.7739713780921003|1.2681072652159826|-0.14264661986123855|-0.08646276453469218|1.1357663877134403| 0.05142487299813041|
      ...
      |G1200860|[0.55134697184525...|[0.84536048028294...|0.8453604802829415|1.1649733636908406|  -0.251498842124316|-0.05398789321041379|1.1379273562164913|0.006627433190429924|
      |G1200860|[0.55033421105934...|[0.84497922527196...|0.8449792252719691|1.1720647495314305| -0.2684042852950131|-0.01816121020906...|1.1458699947581152|0.009816855921066535|
      |G1200860|[0.55235973263115...|[0.83180258697776...|0.8318025869777639|1.1888588297495093|  -0.247244637606236|-0.01792134094207...|1.1359619004162465| 0.00902030342487202|
      |G1200860|[0.55073931537370...|[0.84546368981010...|0.8454636898101023|1.1706852781462835| -0.2708025633589945|-0.00719615226018...|1.1392678220552446|0.008471343752829472|
      |G1200860|[0.54911889811626...|[0.84514470930467...|0.8451447093046771|1.1706297920180826| -0.2708819367777809|-0.00756457734791...|1.1372865563117458|0.014087994825427497|
      ...
      +--------+--------------------+--------------------+------------------+------------------+--------------------+--------------------+------------------+--------------------+
     */
    val pca: PrincipleComponentAnalysis = new PrincipleComponentAnalysis()
    var pcaDF: Dataset[Row] = pca.runPCA(sparkConnector, pcaFeatures)

    /*
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
    pcaDF = pcaDF.groupBy(col("gis_join")).agg(
      avg("pcaFeatures_0").as("avg_pc_0"),
      avg("pcaFeatures_1").as("avg_pc_1"),
      avg("pcaFeatures_2").as("avg_pc_2"),
      avg("pcaFeatures_3").as("avg_pc_3"),
      avg("pcaFeatures_4").as("avg_pc_4"),
      avg("pcaFeatures_5").as("avg_pc_5")
    ).select("gis_join", "avg_pc_0", "avg_pc_1", "avg_pc_2", "avg_pc_3", "avg_pc_4", "avg_pc_5")

    val kMeansClustering: KMeansClustering = new KMeansClustering()
    kMeansClustering.runClustering(sparkConnector, pcaDF, clusteringFeatures, clusteringK)
  }

  /**
   * Writes the modeling header to a CSV file
   */
  def writeSequentialModelHeader(filename: String): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File(filename)
      )
    )
    bw.write("gis_join,time_ms,iterations,rmse")
    bw.close()
  }

  /**
   * Writes the modeling stats for a single model to a CSV file
   */
  def writeSequentialModelStats(filename: String, gisJoin: String, time: Long, rmse: Double, iterations: Int, numRecords: Long): Unit = {
    val bw = new BufferedWriter(
      new FileWriter(
        new File(filename),
        true
      )
    )
    bw.write("%s,%d,%d,%d,%f\n".format(gisJoin, time, numRecords, iterations, rmse))
    bw.close()
  }

}
