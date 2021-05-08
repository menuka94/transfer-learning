package org.sustain

import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset, Row, RowFactory}
import org.apache.spark.sql.functions.{avg, col, collect_list, min, row_number, struct}
import org.apache.spark.ml.regression.LinearRegression
import com.mongodb.spark.MongoSpark
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorDisassembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Dataset, Row, SparkSession}

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

@SerialVersionUID(114L)
class Experiment() extends Serializable {

  def transferLearning(sparkMaster: String, appName: String, mongosRouters: Array[String], mongoPort: String,
                       database: String, collection: String, clusteringFeatures: Array[String], clusteringK: Int, regressionFeatures: Array[String],
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
      |G1200870|[0.55600567146040...|[0.68385808112507...|0.6838580811250755|1.4102913185089028| 0.02880129713901572|-0.20988835643907133| 1.220484723051305| 0.07826483591938016|
      |G1200870|[0.55772736479643...|[0.66660908921372...|0.6666090892137281| 1.432883670962919| 0.04246480131356592|-0.18732207119830727| 1.203584563727429|  0.0753642306249851|
      |G1200870|[0.55499291067449...|[0.75565165269005...| 0.755651652690051|1.2957279183963304| -0.1079763875543191|-0.11465980130696433|1.1529245450749506|  0.0607012336436967|
      |G1200870|[0.55468908243872...|[0.72406202099240...|0.7240620209924077|1.3583191193657465|-0.06203154074853057|-0.11214500537568343| 1.194904342391342|  0.0740213808764568|
      |G1200870|[0.55701843224630...|[0.70259295830020...|0.7025929583002002|1.3703079793806712|-0.01583567910749678| -0.1675068608535143|1.1673259442345167| 0.06516650439735809|
      |G1200870|[0.55894267773951...|[0.65973792208689...|0.6597379220868983|1.4284477499882495|0.061121245449639955|-0.21750232376999368| 1.174309569259993| 0.06709715077275447|
      |G1200870|[0.55752481263925...|[0.70411932843546...|0.7041193284354648|1.3649629523219389|-0.01802379297813...|-0.16591462749080002|1.1542440227560387| 0.06492486683775185|
      |G1200870|[0.55600567146040...|[0.72890749027291...|0.7289074902729173|1.3314318564116634|-0.06278532010811427|-0.13755236800371418|1.1529930464392448| 0.06649874375933865|
      |G1200870|[0.55549929106744...|[0.73739245072801...|0.7373924507280125|1.3213315576102436| -0.0757517891947212|-0.13403755765966915|1.1606796440694302| 0.06347218008862499|
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





    // Create LR models for cluster centroid GISJoins
    /*
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

     */

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

    /*
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

    // --- DEBUGGING ---
    println("\n\n>>> MODEL QUEUES <<<\n")
    clustersQueues.foreach{ println }


    try {
      clustersQueues.foreach(queue => queue.start())
      clustersQueues.foreach(queue => queue.wait())
    } catch {
      case e: java.lang.IllegalMonitorStateException => println("\n\nn>>>Caught IllegalMonitorStateException!")
    }

     */
  }

}
