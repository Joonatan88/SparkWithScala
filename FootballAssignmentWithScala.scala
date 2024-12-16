// Databricks notebook source
//import statements for the entire notebook
//add anything that is required here

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

// COMMAND ----------

//Data initalization

val path = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/events.parquet"
val eventDF: DataFrame = spark.read.parquet(path)

// COMMAND ----------

//calculating match results

import org.apache.spark.sql.functions.array_contains

val goalsDF = eventDF
  .filter(col("event") === "Save attempt" && array_contains(col("tags"), "Goal"))
  .withColumn("homeTeamGoals", when(col("eventTeam") === col("awayTeam"), 1).otherwise(0))
  .withColumn("awayTeamGoals", when(col("eventTeam") === col("homeTeam"), 1).otherwise(0))
  .groupBy("matchId","competition", "season", "homeTeam", "awayTeam")
  .agg(
    sum("homeTeamGoals").alias("homeTeamGoals"),
    sum("awayTeamGoals").alias("awayTeamGoals")
  )

val allGames = eventDF.select("matchId", "competition", "season", "homeTeam", "awayTeam").distinct()

val matchDF: DataFrame = allGames.join(goalsDF, Seq("matchId", "competition", "season", "homeTeam", "awayTeam"), "left")
.na.fill(0, Seq("homeTeamGoals", "awayTeamGoals"))

// COMMAND ----------

//teams points for the season

val home_stats = matchDF.select(
  col("competition"),
  col("season"),
  col("homeTeam").alias("team"),
  col("homeTeamGoals").alias("goals_scored"),
  col("awayTeamGoals").alias("goals_conceded"),
  expr("cast(homeTeamGoals > awayTeamGoals as int)").alias("wins"),
  expr("cast(homeTeamGoals = awayTeamGoals as int)").alias("draws"),
  expr("cast(homeTeamGoals < awayTeamGoals as int)").alias("losses"),
  lit(1).alias("games")
)

val away_stats = matchDF.select(
  col("competition"),
  col("season"),
  col("awayTeam").alias("team"),
  col("awayTeamGoals").alias("goals_scored"),
  col("homeTeamGoals").alias("goals_conceded"),
  expr("cast(awayTeamGoals > homeTeamGoals as int)").alias("wins"),
  expr("cast(awayTeamGoals = homeTeamGoals as int)").alias("draws"),
  expr("cast(awayTeamGoals < homeTeamGoals as int)").alias("losses"),
  lit(1).alias("games")
)

val teamStatsDF = home_stats.union(away_stats)

val seasonDF: DataFrame = teamStatsDF.groupBy("competition", "season", "team").agg(
  sum("games").as("games"),
  sum("wins").as("wins"),
  sum("draws").as("draws"),
  sum("losses").as("losses"),
  sum("goals_scored").as("goalsScored"),
  sum("goals_conceded").as("goalsConceded"),
  (sum("wins") * 3 + sum("draws")).as("points")
)

// COMMAND ----------

//Table for Premier League

val premDF = seasonDF.filter(col("competition") === "English Premier League")
  .groupBy("team")
  .agg(
    sum("games").alias("Pld"),
    sum("wins").alias("W"),
    sum("draws").alias("D"),
    sum("losses").alias("L"),
    sum("goalsScored").alias("GF"),
    sum("goalsConceded").alias("GA"),
    (sum("goalsScored") - sum("goalsConceded")).alias("GD"),
    sum("points").alias("Pts")
  ).orderBy(desc("Pts"), desc("GD"), desc("GF"))

  val premDFwithPosAndGD = premDF.withColumn("Pos", monotonically_increasing_id() + 1)
  .withColumn("GD", 
    when(col("GD") >= 0, concat(lit("+"), col("GD").cast("string")))
    .otherwise(col("GD").cast("string"))
  )


val englandDF: DataFrame = premDFwithPosAndGD.select(
  col("Pos"),
  col("team").alias("Team"),
  col("Pld"),
  col("W"),
  col("D"),
  col("L"),
  col("GF"),
  col("GA"),
  col("GD"),
  col("Pts")
)

println("English Premier League table for season 2017-2018")
englandDF.show(20, false)

// COMMAND ----------

//Calculating number of passes

val passesDF = eventDF
  .filter(col("event") === "Pass")
  .withColumn("succesfulPasses", when(array_contains(col("tags"), "Accurate"), 1).otherwise(0))
  .withColumn("totalPasses", lit(1))
  
val homeTeamPassesDF = passesDF
  .filter(col("eventTeam") === col("homeTeam"))
  .groupBy("matchId", "homeTeam", "competition", "season")
  .agg(
    sum("succesfulPasses").alias("succesfulPasses"),
    sum("totalPasses").alias("totalPasses")
  ).withColumnRenamed("homeTeam", "team")

val awayTeamPassesDF = passesDF
  .filter(col("eventTeam") === col("awayTeam"))
  .groupBy("matchId", "awayTeam", "competition", "season")
  .agg(
    sum("succesfulPasses").alias("succesfulPasses"),
    sum("totalPasses").alias("totalPasses")
  ).withColumnRenamed("awayTeam", "team")

val matchPassDF: DataFrame = homeTeamPassesDF.union(awayTeamPassesDF)

display(matchPassDF)


// COMMAND ----------

//Query for teams with the worst passing accuracy

import org.apache.spark.sql.expressions.Window

val passratioForSeasonDF = matchPassDF
  .withColumn("passSuccessRatio", round((col("succesfulPasses") / col("totalPasses")) * 100, 2))
  .groupBy("competition", "team")
  .agg(
    avg("passSuccessRatio").alias("passSuccessRatio")
  )
  .withColumn("passSuccessRatio", round(col("passSuccessRatio"), 2))

val window = Window.partitionBy("competition").orderBy("passSuccessRatio")
val rankedDF = passratioForSeasonDF.withColumn("rank", row_number().over(window))

val lowestPassSuccessRatioDF: DataFrame = rankedDF.filter(col("rank") === 1).drop("rank").orderBy(asc("passSuccessRatio"))

println("The teams with the lowest ratios for successful passes for each league in season 2017-2018:")
lowestPassSuccessRatioDF.show(5, false)

// COMMAND ----------

//Query for top 2 best teams from each league

val allTeamsDF = seasonDF
  .groupBy("team", "competition")
  .agg(
    sum("games").alias("Pld"),
    sum("wins").alias("W"),
    sum("draws").alias("D"),
    sum("losses").alias("L"),
    sum("goalsScored").alias("GF"),
    sum("goalsConceded").alias("GA"),
    (sum("goalsScored") - sum("goalsConceded")).alias("GD"),
    sum("points").alias("Pts")
  ).orderBy(desc("Pts"), desc("GD"), desc("GF"))

val passratioForSeasonRenamedDF = passratioForSeasonDF.withColumnRenamed("competition", "competition_pass") //Fixes issue in joining 2 dfs

val allTeamsWithPassRatioDF = allTeamsDF.join(passratioForSeasonRenamedDF, Seq("team"), "left")
  .select(
    col("team"),
    col("competition"),
    col("Pld"),
    col("W"),
    col("D"),
    col("L"),
    col("GF"),
    col("GA"),
    col("GD"),
    col("Pts"),
    col("passSuccessRatio")
  ).orderBy(col("Pts"))

val windowSpec = Window.partitionBy("competition").orderBy(desc("Pts"), desc("GD"), desc("GF"))
val allTeamsWithPosDF = allTeamsWithPassRatioDF.withColumn("Pos", row_number().over(windowSpec))

val allTeamsWithPtsAvgDF = allTeamsWithPosDF.withColumn("Avg", round(col("Pts") / col("Pld"), 2))
  .withColumn("GD", 
    when(col("GD") >= 0, concat(lit("+"), col("GD").cast("string"))) //assigns + to positive GD
    .otherwise(col("GD").cast("string"))
  )
  .select(
    col("team"),
    col("competition"),
    col("Pos"),
    col("Pld"),
    col("W"),
    col("D"),
    col("L"),
    col("GF"),
    col("GA"),
    col("GD"),
    col("Pts"),
    col("Avg"),
    col("passSuccessRatio").alias("PassRatio")
  ).orderBy(desc("Pts"))

  val bestDF: DataFrame = allTeamsWithPtsAvgDF.filter(col("Pos") <= 2 )
  .orderBy(desc("Avg"))
  .limit(10)

println("The top 2 teams for each league in season 2017-2018")
bestDF.show(10, false)