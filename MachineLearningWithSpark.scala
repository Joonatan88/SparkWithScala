// Databricks notebook source
// MAGIC %md
// MAGIC # Preview
// MAGIC This notebook involves experimenting with the classifiers provided by the Spark machine learning library. Time series data collected in the ProCem research (https://www.senecc.fi/projects/procem-2) project is used as the training and test data.
// MAGIC
// MAGIC
// MAGIC ### Steps:
// MAGIC ### 
// MAGIC 1. The first task is to train and test a machine learning model with Random forest classifier in six different cases: 
// MAGIC -   Predict the month (1-12) using the three weather measurements (temperature, humidity, and wind speed) as input   
// MAGIC -   Predict the month (1-12) using the three power measurements (tenants, maintenance, and solar panels) as input  
// MAGIC -   Predict the month (1-12) using all seven measurements (weather values, power values, and price) as input 
// MAGIC -   Predict the hour of the day (0-23) using the three weather measurements (temperature, humidity, and wind speed) as input
// MAGIC -   Predict the hour of the day (0-23) using the three power measurements (tenants, maintenance, and solar panels) as input
// MAGIC -   Predict the hour of the day (0-23) using all seven measurements (weather values, power values, and price) as input
// MAGIC
// MAGIC 2. The second task is to create a Naive-Bayes model and compare the results to Random forest classifier model on slightly different measures

// COMMAND ----------

// MAGIC %md
// MAGIC #### Data description
// MAGIC
// MAGIC The dataset contains time series data from a period of 13 months (from the beginning of May 2023 to the end of May 2024). Each row contains the average of the measured values for a single minute. The following columns are included in the data:
// MAGIC
// MAGIC | column name        | column type   | description |
// MAGIC | ------------------ | ------------- | ----------- |
// MAGIC | time               | long          | The UNIX timestamp in second precision |
// MAGIC | temperature        | double        | The temperature measured by the weather station on top of Sähkötalo (`°C`) |
// MAGIC | humidity           | double        | The humidity measured by the weather station on top of Sähkötalo (`%`) |
// MAGIC | wind_speed         | double        | The wind speed measured by the weather station on top of Sähkötalo (`m/s`) |
// MAGIC | power_tenants      | double        | The total combined electricity power used by the tenants on Kampusareena (`W`) |
// MAGIC | power_maintenance  | double        | The total combined electricity power used by the building maintenance systems on Kampusareena (`W`) |
// MAGIC | power_solar_panels | double        | The total electricity power produced by the solar panels on Kampusareena (`W`) |
// MAGIC | electricity_price  | double        | The market price for electricity in Finland (`€/MWh`) |
// MAGIC

// COMMAND ----------

//Data preparation
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

val MlDataPath = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/energy/procem_13m.parquet/procem.parquet"
val MLRawDataDF: DataFrame = spark.read.parquet(MlDataPath)

val MLCleanedDataDF = MLRawDataDF.na.drop()

val MLdataWithTimeDF = MLCleanedDataDF.withColumn("month", month(from_unixtime(col("time"))))
  .withColumn("hour", hour(from_unixtime(col("time"))))

// COMMAND ----------

// MAGIC %md
// MAGIC ## Task 1
// MAGIC Creating a Random Forest Classifier model to predict hours of day and month from different measures.

// COMMAND ----------

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{StructType, StructField, StringType, DoubleType}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.udf

// RF MODEL CREATION 

def Random_forest(data: DataFrame, features: Array[String], labels: String): (String, String, String, Double, Double, Double, Double) = {

  val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 1)

  val featureAssembler = new VectorAssembler().setInputCols(features).setOutputCol("features")

  val labelIndexer = new StringIndexer().setInputCol(labels).setOutputCol("indexedLabel").fit(data)

  val randomForest = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("indexedLabel").setNumTrees(8)

  val pipeline = new Pipeline().setStages(Array(featureAssembler, labelIndexer, randomForest))

  //training
  println(s"Training a 'RandomForest' model to predict $labels based on inputs: ${features.mkString(" ")}")
  val model = pipeline.fit(trainingData)

  //predictions
  val predictions = model.transform(testData)

  //built-in evaluator
  val sparkEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  
  val accuracy = sparkEvaluator.evaluate(predictions)
  println(s"Accuracy: $accuracy")

  //Custom Evaluations
  val correctPredictions = predictions.filter(col("indexedLabel") === col("prediction")).count()
  val totalPredictions = predictions.count()
  val customAccuracy = correctPredictions.toDouble / totalPredictions
  println(s"Custom Accuracy: $customAccuracy")


  val cyclicRange = if (labels == "month") 12 else 24

  val oneAway = predictions.filter(row => {
    val label = row.getAs[Double]("indexedLabel")
    val pred = row.getAs[Double]("prediction")
    Math.abs(label - pred) <= 1 || Math.abs(label - pred) == cyclicRange - 1
  }).count().toDouble / totalPredictions

  val twoAway = predictions.filter(row => {
    val label = row.getAs[Double]("indexedLabel")
    val pred = row.getAs[Double]("prediction")
    Math.abs(label - pred) <= 2 || Math.abs(label - pred) >= cyclicRange - 2
  }).count().toDouble / totalPredictions
  
  println(s"Percentage within 1 unit: $oneAway")
  println(s"Percentage within 2 units: $twoAway")

  val extractProbability = udf((probability: Vector, label: Double) => {
  val labelIndex = label.toInt
  probability(labelIndex)
  })

  // Add the correct probability column
  val predictionsWithProb = predictions.withColumn(
    "correctProbability",
    extractProbability(col("probability"), col("indexedLabel"))
  )

  // Aggregate the average probability for correct predictions
  val avgProbability = predictionsWithProb.agg(avg("correctProbability")).first().getDouble(0)
  println(s"Average Probability for Correct Value: $avgProbability")

    // Return the results
    ("RandomForest", features.mkString(" "), labels, customAccuracy * 100, oneAway * 100, twoAway * 100, avgProbability)
  }

// COMMAND ----------

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

//Schema for adding results from model to a DataFrame
val MLschema = StructType(Array(
  StructField("classifier", StringType, true),
  StructField("input", StringType, true),
  StructField("label", StringType, true),
  StructField("correct", DoubleType, true),
  StructField("within_one", DoubleType, true),
  StructField("within_two", DoubleType, true),
  StructField("avgProbability", DoubleType, true)
))

//Model calculating results for different measurements from data
val RFresults = Seq(
  Random_forest(MLdataWithTimeDF, Array("temperature", "humidity", "wind_speed", "power_tenants", "power_maintenance", "power_solar_panels", "electricity_price"), "month"),
  Random_forest(MLdataWithTimeDF, Array("temperature", "humidity", "wind_speed"), "month"),
  Random_forest(MLdataWithTimeDF, Array("power_tenants", "power_maintenance", "power_solar_panels"), "month"),
  Random_forest(MLdataWithTimeDF, Array("temperature", "humidity", "wind_speed", "power_tenants", "power_maintenance", "power_solar_panels", "electricity_price"), "hour"),
  Random_forest(MLdataWithTimeDF, Array("power_tenants", "power_maintenance", "power_solar_panels"), "hour"),
  Random_forest(MLdataWithTimeDF, Array("temperature", "humidity", "wind_speed"), "hour")
)

//Moving results from RF model to rows to be added to dataframe
val rows = RFresults.map { case (classifier, input, label, correct, within_one, within_two, avgProbability) =>
  Row(classifier, input, label, correct, within_one, within_two, avgProbability)
}

//Models results to dataframe
val MLresultsDF = spark.createDataFrame(spark.sparkContext.parallelize(rows), MLschema)

//Query for getting results
val MLFinalDF = MLresultsDF.select(
  col("classifier"),
  col("input"),
  col("label"),
  round(col("correct"), 2).alias("correct"),
  round(col("within_one"), 2).alias("within_one"),
  round(col("within_two"), 2).alias("within_two"),
  round(col("avgProbability"), 4).alias("avg_prob")
)

display(MLFinalDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Task2
// MAGIC
// MAGIC Naive-Bayes vs Random Forest classifier

// COMMAND ----------

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{StructType, StructField, StringType, DoubleType}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.udf

//Naive-Bayes model

def Naive_Bayes(data: DataFrame, features: Array[String], labels: String): (String, String, String, Double, Double, Double, Double) = {

  val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 1)

  val featureAssembler = new VectorAssembler()
    .setInputCols(features)
    .setOutputCol("features")

  val labelIndexer = new StringIndexer()
    .setInputCol(labels)
    .setOutputCol("indexedLabel")
    .fit(data)

  val naiveBayes = new NaiveBayes()
    .setFeaturesCol("features")
    .setLabelCol("indexedLabel")

  val pipeline = new Pipeline().setStages(Array(featureAssembler, labelIndexer, naiveBayes))

  //Training
  println(s"Training a 'NaiveBayes' model to predict $labels based on inputs: ${features.mkString(" ")}")
  val model = pipeline.fit(trainingData)

  // Make predictions on the test set
  val predictions = model.transform(testData)

  //built-in eval
  val sparkEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  
  val accuracy = sparkEvaluator.evaluate(predictions)
  println(s"Accuracy: $accuracy")

  //Custom Evaluations
  val correctPredictions = predictions.filter(col("indexedLabel") === col("prediction")).count()
  val totalPredictions = predictions.count()
  val customAccuracy = correctPredictions.toDouble / totalPredictions
  println(s"Custom Accuracy: $customAccuracy")

  val cyclicRange = if (labels == "month") 12 else 24

  val oneAway = predictions.filter(row => {
    val label = row.getAs[Double]("indexedLabel")
    val pred = row.getAs[Double]("prediction")
    Math.abs(label - pred) <= 1 || Math.abs(label - pred) == cyclicRange - 1
  }).count().toDouble / totalPredictions

  val twoAway = predictions.filter(row => {
    val label = row.getAs[Double]("indexedLabel")
    val pred = row.getAs[Double]("prediction")
    Math.abs(label - pred) <= 2 || Math.abs(label - pred) >= cyclicRange - 2
  }).count().toDouble / totalPredictions
  
  println(s"Percentage within 1 unit: $oneAway")
  println(s"Percentage within 2 units: $twoAway")

  val extractProbability = udf((probability: Vector, label: Double) => {
    val labelIndex = label.toInt
    probability(labelIndex)
  })

  val predictionsWithProb = predictions.withColumn(
    "correctProbability",
    extractProbability(col("probability"), col("indexedLabel"))
  )

  val avgProbability = predictionsWithProb.agg(avg("correctProbability")).first().getDouble(0)
  println(s"Average Probability for Correct Value: $avgProbability")

  ("NaiveBayes", features.mkString(" "), labels, customAccuracy * 100, oneAway * 100, twoAway * 100, avgProbability)
}


// COMMAND ----------

// MAGIC %md
// MAGIC We compare Naive-Bayes models performance to Random Forest classifiers performance on predicting the day of week from: temperature, humidity and windspeed.

// COMMAND ----------

import org.apache.spark.sql.types.{StructType, StructField, StringType, DoubleType}
import org.apache.spark.sql.functions.{col, date_format, from_unixtime, round}

val MLdataWithDayDF = MLCleanedDataDF.withColumn("WeekDay", date_format(from_unixtime(col("time")), "EEEE"))

//Fixes a bug
val filteredMLdataWithDayDF = MLdataWithDayDF.filter(
  col("temperature") >= 0 && 
  col("humidity") >= 0 && 
  col("wind_speed") >= 0 &&
  !col("temperature").isNaN && 
  !col("humidity").isNaN && 
  !col("wind_speed").isNaN &&
  !col("temperature").isNull && 
  !col("humidity").isNull && 
  !col("wind_speed").isNull
)

//Schema initalization
val MLschema = StructType(Array(
  StructField("classifier", StringType, true),
  StructField("input", StringType, true),
  StructField("label", StringType, true),
  StructField("correct", DoubleType, true),
  StructField("within_one", DoubleType, true),
  StructField("within_two", DoubleType, true),
  StructField("avgProbability", DoubleType, true)
))

//Naive-Bayes results
val NBresults = Seq(
  Naive_Bayes(filteredMLdataWithDayDF,  Array("temperature", "humidity", "wind_speed"), "WeekDay")
)

//Naive-Bayes results to rows
val NBrows = NBresults.map { case (classifier, input, label, correct, within_one, within_two, avgProbability) =>
  Row(classifier, input, label, correct, within_one, within_two, avgProbability)
}

//Naive-Bayes results to dataframe
val NBresultsDF = spark.createDataFrame(spark.sparkContext.parallelize(NBrows), MLschema)

//Naive-Bayes query used in comparison
val NBFinalDF = NBresultsDF.select(
  col("classifier"),
  col("input"),
  col("label"),
  round(col("correct"), 2).alias("correct"),
  round(col("within_one"), 2).alias("within_one"),
  round(col("within_two"), 2).alias("within_two"),
  round(col("avgProbability"), 4).alias("avg_prob")
)

//Results for Random Forest model
val RFresults = Seq(
  Random_forest(filteredMLdataWithDayDF,  Array("temperature", "humidity", "wind_speed"), "WeekDay")
)

//RF results to rows
val RFrows = RFresults.map { case (classifier, input, label, correct, within_one, within_two, avgProbability) =>
  Row(classifier, input, label, correct, within_one, within_two, avgProbability)
}

//Rows to dataframe
val RFresultsDF = spark.createDataFrame(spark.sparkContext.parallelize(RFrows), MLschema)

//Query from dataframe for end result
val RFfinalDF = RFresultsDF.select(
  col("classifier"),
  col("input"),
  col("label"),
  round(col("correct"), 2).alias("correct"),
  round(col("within_one"), 2).alias("within_one"),
  round(col("within_two"), 2).alias("within_two"),
  round(col("avgProbability"), 4).alias("avg_prob")
)

//Comparing RandomForest and NaiveBayes
val LastOneDF = NBFinalDF.union(RFfinalDF)

display(LastOneDF)