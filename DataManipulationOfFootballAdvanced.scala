// Databricks notebook source
// MAGIC %md
// MAGIC Tasks:
// MAGIC
// MAGIC The target of the task is to use the football event data and the additional datasets to determine the following:
// MAGIC
// MAGIC The players with the most total minutes played in season 2017-2018 for each player role
// MAGIC
// MAGIC
// MAGIC The players with higher than +65 for the total plus-minus statistics in season 2017-2018

// COMMAND ----------

//Data prep

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

val path = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/events.parquet"
val eventDF: DataFrame = spark.read.parquet(path)

val path_matches = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/matches.parquet"
val matchesDF: DataFrame = spark.read.parquet(path_matches)

val path_players = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/players.parquet"
val playersDF: DataFrame = spark.read.parquet(path_players)

// COMMAND ----------

//Interrim querying

val secondHalfEventsDF = eventDF.filter(col("eventPeriod") === "2H")

val maxEventTimeDF = secondHalfEventsDF.groupBy("matchId")
  .agg(
    max("eventTime").alias("lastEvent")
  )

val matchLengthDF = maxEventTimeDF.
withColumn("matchLength", ceil(col("lastEvent") / 60 + 45))
.drop(col("lastEvent"))

val matches_withTimeDF = matchesDF.join(matchLengthDF, Seq("matchId"))

// COMMAND ----------

//Interim Querying for played minutes

val homeSubsDF = matches_withTimeDF.select(
  col("matchId"), 
  col("competition"),
  col("season"),
  col("homeTeamData.team").alias("playerTeam"),
  col("matchLength"),
  explode(array(
    col("homeTeamData.substitution1"), 
    col("homeTeamData.substitution2"), 
    col("homeTeamData.substitution3")
  )).alias("sub")
).filter(col("sub").isNotNull)
.select(
  col("matchId"), 
  col("sub.playerIn").alias("playerId"), 
  col("sub.playerOut").alias("playerOutId"),
  col("competition"),
  col("season"),
  col("playerTeam"),
  col("sub.minute").alias("subMinute"), 
  col("matchLength")
)

val awaySubsDF = matches_withTimeDF.select(
  col("matchId"), 
  col("competition"),
  col("season"),
  col("awayTeamData.team").alias("playerTeam"),
  col("matchLength"),
  explode(array(
    col("awayTeamData.substitution1"), 
    col("awayTeamData.substitution2"), 
    col("awayTeamData.substitution3")
  )).alias("sub")
).filter(col("sub").isNotNull)
.select(
  col("matchId"), 
  col("sub.playerIn").alias("playerId"), 
  col("sub.playerOut").alias("playerOutId"),
  col("competition"),
  col("season"),
  col("playerTeam"),
  col("sub.minute").alias("subMinute"), 
  col("matchLength")
)

val subsDF = homeSubsDF.union(awaySubsDF)

val homeLineupDF = matches_withTimeDF.select(
  col("matchId"), 
  col("competition"), 
  col("season"), 
  col("homeTeamData.team").alias("playerTeam"),
  explode(col("homeTeamData.lineup")).alias("playerId"),
  col("matchLength")
).withColumn("startMinute", lit(0))

val awayLineupDF = matches_withTimeDF.select(
  col("matchId"), 
  col("competition"), 
  col("season"), 
  col("awayTeamData.team").alias("playerTeam"),
  explode(col("awayTeamData.lineup")).alias("playerId"),
  col("matchLength")
).withColumn("startMinute", lit(0))

val lineupDF = homeLineupDF.union(awayLineupDF)

val playerEventsDF = lineupDF.join(
  subsDF.select("matchId", "playerOutId", "subMinute").withColumnRenamed("playerOutId", "playerId"), 
  Seq("matchId", "playerId"), "left"
).withColumn("endMinute", when(col("subMinute").isNotNull, col("subMinute")).otherwise(col("matchLength")))
.withColumn("minutes", col("endMinute") - col("startMinute"))
.select(
  col("matchId"), 
  col("playerId"), 
  col("competition"), 
  col("season"), 
  col("playerTeam"), 
  col("startMinute"), 
  col("endMinute"), 
  col("minutes")
)

val combinedDF = playerEventsDF.union(
  subsDF.select(
    col("matchId"), 
    col("playerId"), 
    col("competition"), 
    col("season"), 
    col("playerTeam"), 
    col("subMinute").alias("startMinute"), 
    col("matchLength").alias("endMinute")
  ).withColumn("minutes", col("endMinute") - col("startMinute"))
)


// COMMAND ----------

//Using window function to partition to roles

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.col

val playerTotalMinutes = combinedDF.groupBy("playerId").sum("minutes")

val playersInfoWithMinutesDF = playersDF.join(playerTotalMinutes, Seq("playerId"), "left" )

val windowSpec = Window.partitionBy("role").orderBy(col("sum(minutes)").desc)


// COMMAND ----------

//Final Query for players with most minutes in each role (Goalkeeper, Defender, Midfielder, Attacker)

val mostMinutesDF: DataFrame = playersInfoWithMinutesDF
  .withColumn("rank", row_number().over(windowSpec))
  .withColumn("player", concat_ws(" ", col("firstname"), col("lastName")))
  .filter(col("rank") === 1)
  .drop("rank")
  .select(
    col("role"),
    col("player"),
    col("birthArea"),
    col("sum(minutes)").alias("minutes")
  ).orderBy(desc("minutes"))

println("The players with the most minutes played in season 2017-2018 for each player role:")
mostMinutesDF.show(false)

// COMMAND ----------

//Querying for plusminus

val failedSaveAttemptsDF = eventDF
  .filter(col("event") === "Save attempt" && array_contains(col("tags"), "Goal"))
  .withColumn("goalMinute", when(col("eventPeriod") === "1H", ceil(col("eventTime") / 60)).otherwise(ceil(col("eventTime") / 60 + 45)))
  .withColumn("scoringTeam", when(col("eventTeam") === col("homeTeam"), col("awayTeam")).otherwise(col("homeTeam")))
  .withColumn("goal", lit(1))

val plusMinusDF = failedSaveAttemptsDF.join(combinedDF, "matchId")
  .filter(col("goalMinute").between(col("startMinute"), col("endMinute")))
  .withColumn("plusMinus", when(col("scoringTeam") === col("playerTeam"), 1).otherwise(-1))
  .groupBy("playerId")
  .agg(sum("plusMinus").alias("plusMinus"))

val bestPlayersDF = playersDF.join(plusMinusDF, Seq("playerId"))

// COMMAND ----------

//Top players with +65 plusMinus score

val topPlayers: DataFrame = bestPlayersDF.filter(col("plusMinus") > 65)
  .withColumn("player", concat_ws(" ", col("firstname"), col("lastName")))
  .select(
    col("player"),
    col("birthArea"),
    col("role"),
    col("plusMinus")
  ).orderBy(desc("plusMinus"), asc("birthArea"))

println("The players with higher than +65 for the plus-minus statistics in season 2017-2018:")
topPlayers.show(false)