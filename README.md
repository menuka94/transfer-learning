# transfer-learning

This project is written in Scala, and is Scala Build Tool (SBT) compliant. Documentation provided assumes prior experience with developing Spark applications in Scala, as well as some machine learning knowledge.

## Description

Runs Spark PCA modeling on [NOAA's NAM dataset](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/north-american-mesoscale-forecast-system-nam), followed by K-Means Clustering to get **K** clusters. The cluster centroids are computed by finding the GISJoins which have the smallest squared distance based on principle components. Finally, the application outputs a list of <cluster_id>,<gis_join>,<is_center> in CSV format.

## Usage

To run the [Spark shell](https://spark.apache.org/docs/latest/quick-start.html#interactive-analysis-with-the-spark-shell) with all required dependencies for development/debugging:
- `./run_shell.sh`

To run the application via [spark-submit](https://spark.apache.org/docs/latest/submitting-applications.html):
- `./run.sh`
