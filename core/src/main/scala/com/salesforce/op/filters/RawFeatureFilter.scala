/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.filters

import scala.math.{abs, min}

import com.salesforce.op.OpParams
import com.salesforce.op.features.types._
import com.salesforce.op.features.{OPFeature, TransientFeature}
import com.salesforce.op.readers.{DataFrameFieldNames, Reader}
import com.salesforce.op.stages.impl.feature.{HashAlgorithm, Inclusion, NumericBucketizer, TextTokenizer}
import com.salesforce.op.stages.impl.preparators.CorrelationType
import com.salesforce.op.utils.spark.RichRow._
import com.twitter.algebird.Monoid
import com.twitter.algebird.Semigroup
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.{Matrix, Vector}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.slf4j.LoggerFactory

/**
 * Specialized stage that will load up data and compute distributions and empty counts on raw features.
 * This information is then used to compute which raw features should be excluded from the workflow DAG
 * @param trainingReader reader to get the training data
 * @param scoreReader reader to get the scoring data for comparison (optional - if not present will exclude based on
 *                    training data features only)
 * @param bins number of bins to use in computing feature distributions (histograms for numerics, hashes for strings)
 * @param minFill minimum fill rate a feature must have in the training dataset and scoring dataset to be kept
 * @param maxFillDifference maximum acceptable fill rate difference between training and scoring data to be kept
 * @param maxFillRatioDiff maximum acceptable fill ratio between training and scoring (larger / smaller)
 * @param maxJSDivergence maximum Jensen-Shannon divergence between training and scoring distributions to be kept
 * @param maxCorrelation maximum absolute correlation allowed between raw predictor null indicator and label
 * @param correlationType type of correlation metric to use
 * @param jsDivergenceProtectedFeatures features that are protected from removal by JS divergence check
 * @param protectedFeatures features that are protected from removal
 * @tparam T datatype of the reader
 */
class RawFeatureFilter[T]
(
  val trainingReader: Reader[T],
  val scoreReader: Option[Reader[T]],
  val minFill: Double,
  val maxFillDifference: Double,
  val maxFillRatioDiff: Double,
  val maxCJSDivergence: Double,
  val maxJSDivergence: Double,
  val maxCorrelation: Double,
  val correlationType: CorrelationType = CorrelationType.Pearson,
  val cjsDivergenceProtectedFeatures: Set[String] = Set.empty,
  val jsDivergenceProtectedFeatures: Set[String] = Set.empty,
  val protectedFeatures: Set[String] = Set.empty,
  val maxBins: Int = FeatureDistribution.DefaultMaxBins
) extends Serializable {

  assert(minFill >= 0.0 && minFill <= 1.0, s"Invalid minFill size $minFill, minFill must be between 0 and 1")
  assert(maxFillDifference >= 0.0 && maxFillDifference <= 1.0, s"Invalid maxFillDifference size $maxFillDifference," +
    s" maxFillDifference must be between 0 and 1")
  assert(maxFillRatioDiff >= 0.0, s"Invalid maxFillRatioDiff size $maxFillRatioDiff," +
    s" maxFillRatioDiff must be greater than 0.0")
  assert(maxJSDivergence >= 0.0 && maxJSDivergence <= 1.0, s"Invalid maxJSDivergence size $maxJSDivergence," +
    s" maxJSDivergence must be between 0 and 1")

  @transient protected lazy val log = LoggerFactory.getLogger(this.getClass)

  private val hasher: HashingTF = new HashingTF(numFeatures = maxBins)
    .setBinary(false)
    .setHashAlgorithm(HashAlgorithm.MurMur3.toString.toLowerCase)

  /**
   * Get binned counts of the feature distribution and empty count for each raw feature
   * @param data data frame to compute counts on
   * @param features list of raw, non-protected, features contained in the dataframe
   * @param allFeatureInfo existing feature info to use
   * @return a sequence of distribution summaries for each raw feature
   */
  private[op] def computeFeatureStats(
    data: DataFrame,
    features: Array[OPFeature],
    allFeatureInfo: Option[AllFeatureInformation] = None): AllFeatureInformation = {
    val (responses, predictors): (Array[TransientFeature], Array[TransientFeature]) = {
      val (allResponses, allPredictors) = features.partition(_.isResponse)
      val respOut = allResponses.map(TransientFeature(_)).flatMap {
        case f if f.getFeature().isSubtypeOf[OPNumeric[_]] =>
          log.info("Using numeric response: {}", f.name)
          Option(f)
        case f =>
          log.info("Not using non-numeric response in raw feature filter: {}", f.name)
          None
      }
      val predOut = allPredictors.map(TransientFeature(_))

      (respOut, predOut)
    }
    val preparedFeatures: RDD[PreparedFeatures] =
      data.rdd.map(PreparedFeatures(_, responses, predictors))

    val (responseDists, predictorDists): (Map[FeatureKey, FeatureDistribution], Map[FeatureKey, FeatureDistribution]) =
      preparedFeatures
        .map(f => f.getFeatureDistributions(hasher = hasher, maxBins = maxBins) -> 1.0)
        .reduce(_ + _) match {
        case ((rdists, pdists), count) =>
          val f: FeatureDistribution => FeatureDistribution = d => d.copy(nullCount = count - d.nullCount)
          rdists.mapValues(f) -> pdists.mapValues(f)
      }
    val (responseKeys, predictorKeys): (Array[FeatureKey], Array[FeatureKey]) =
      responseDists.map(_._1).toArray -> predictorDists.map(_._1).toArray
    val correlationInfo: Map[FeatureKey, Map[FeatureKey, Double]] =
      allFeatureInfo.map(_.correlationInfo).getOrElse {
        val emptyCorr: Map[FeatureKey, Map[FeatureKey, Double]] = Map()
        val corrRDD: RDD[Vector] = preparedFeatures.map(_.getNullLabelLeakageVector(responseKeys, predictorKeys))
        val corrMatrix: Matrix = Statistics.corr(corrRDD, correlationType.sparkName)

        responseKeys.zipWithIndex.map { case (responseKey, i) =>
          responseKey -> predictorKeys.zipWithIndex.map { case (predictorKey, j) =>
            predictorKey -> min(abs(corrMatrix(i, j + responseKeys.length)), 1.0)
          }.toMap
        }.toMap
      }

    AllFeatureInformation(
      responseDistributions = responseDists,
      predictorDistributions = predictorDists,
      correlationInfo = correlationInfo)
  }

  /**
   * Take in the distribution summaries for datasets (scoring summary may be empty) and determine which
   * features should be dropped (including maps with all keys dropped) and which map keys need to be dropped
   * @param trainingDistribs summary of distributions for training data features
   * @param scoringDistribs summary of distributions for scoring data features (may be an empty seq)
   * @param correlationInfo info needed to determine feature to drop based on null label-leakage correlation
   * @return a list of feature names that should be dropped and a map of map keys that should be dropped
   *         Map(featureName -> key)
   */
  private[op] def getFeaturesToExclude(
    trainingDistribs: Map[FeatureKey, FeatureDistribution],
    scoringDistribs: Map[FeatureKey, FeatureDistribution],
    correlationInfo: Map[FeatureKey, Map[FeatureKey, Double]]
  ): (Seq[String], Map[String, Set[String]]) = {
    val trainingReasons: Map[FeatureKey, List[String]] = for {
      trainingPair <- trainingDistribs
      (featureKey, trainingDistrib) = trainingPair
      // scoringDistrib <- scoringDistribs.get(featureKey)
    } yield {
      val trainingUnfilled =
        if (trainingDistrib.fillRate >= minFill) None
        else Option(s"training fill rate did not meet min required ($minFill)")
      val trainingNullLabelLeaker =
        if (correlationInfo.map(_._2.get(featureKey).forall(_ <= maxCorrelation)).forall(identity(_))) None
        else Option(s"null indicator correlation (absolute) exceeded max allowed ($maxCorrelation)")

      featureKey -> List(trainingUnfilled, trainingNullLabelLeaker).flatten
    }
    val distribMismatches: Map[FeatureKey, List[String]] = for {
      scoringPair <- scoringDistribs
      (featureKey, scoringDistrib) = scoringPair
      trainingDistrib <- trainingDistribs.get(featureKey)
    } yield {
      val scoringFillRate = scoringDistrib.fillRate
      val trainingFillRate = trainingDistrib.fillRate
      val cjsDivergence = trainingDistrib.cjsDivergence(scoringDistrib)
      val jsDivergence = trainingDistrib.jsDivergence(scoringDistrib)
      val fillRatioDiff = trainingDistrib.relativeFillRatio(scoringDistrib)
      val fillRateDiff = trainingDistrib.relativeFillRatio(scoringDistrib)

      log.info(s"\nTraining Histogram=${trainingDistrib.histogram}\n" +
        s"Scoring Histogram=${scoringDistrib.histogram}\n" +
        s"Train Fill=$trainingFillRate, Score Fill=$scoringFillRate, " +
        s"CJS Divergence $cjsDivergence, JS Divergence=$jsDivergence, " +
        s"Fill Rate Difference=$fillRateDiff, Fill Ratio Difference=$fillRatioDiff")

      val scoringUnfilled =
        if (scoringDistrib.fillRate >= minFill) None
        else Option(s"scoring fill rate did not meet min required ($minFill)")
      val cjsDivergenceCheck =
        if (cjsDivergenceProtectedFeatures.contains(featureKey._1) || cjsDivergence <= maxJSDivergence) None
        else Option(s"CJS Divergence exceeded max allowed ($maxCJSDivergence)")
      val jsDivergenceCheck =
        if (jsDivergenceProtectedFeatures.contains(featureKey._1) || jsDivergence <= maxJSDivergence) None
        else Option(s"JS Divergence exceeded max allowed ($maxJSDivergence)")
      val fillRateCheck =
        if (fillRateDiff <= maxFillDifference) None
        else Option(s"fill rate difference exceeded max allowed ($maxFillDifference)")
      val fillRatioCheck =
        if (fillRatioDiff <= maxFillRatioDiff) None
        else Option(s"fill ratio difference exceeded max allowed ($maxFillRatioDiff)")

      featureKey ->
        List(scoringUnfilled, cjsDivergenceCheck, jsDivergenceCheck, fillRateCheck, fillRatioCheck).flatten
    }
    val allReasons = trainingReasons + distribMismatches // NOTE: Uses Algebird's Map monoid
    val toDrop: Map[FeatureKey, List[String]] = allReasons.filter(_._2.nonEmpty)
    val toKeep: Map[FeatureKey, List[String]] = allReasons.filter(_._2.isEmpty)

    toDrop.foreach { case (featureKey, reasons) =>
      log.info(s"Dropping feature $featureKey because:\n\t${reasons.mkString("\n\t")}\n")
    }

    val toDropFeatures = toDrop.map(_._1).groupBy(_._1)
    val toKeepFeatures = toKeep.map(_._1).groupBy(_._1)
    val mapKeys = toKeepFeatures.keySet.intersect(toDropFeatures.keySet)
    val toDropNames = toDropFeatures.collect { case (k, _) if !mapKeys.contains(k) => k }.toSeq
    val toDropMapKeys = toDropFeatures.collect { case (k, v) if mapKeys.contains(k) => k -> v.flatMap(_._2).toSet }

    toDropNames -> toDropMapKeys
  }

  /**
   * Function that gets raw features and params used in workflow. Will use this information along with readers for this
   * stage to determine which features should be dropped from the workflow
   * @param rawFeatures raw features used in the workflow
   * @param parameters parameters used in the workflow
   * @param spark spark instance
   * @return dataframe that has had bad features and bad map keys removed and a list of all features that should be
   *         dropped from the DAG
   */
  // TODO return distribution information to attach to features that are kept
  def generateFilteredRaw(rawFeatures: Array[OPFeature], parameters: OpParams)
    (implicit spark: SparkSession): FilteredRawData = {

    val trainData = trainingReader.generateDataFrame(rawFeatures, parameters).persist()
    log.info("Loaded training data")
    assert(trainData.count() > 0, "RawFeatureFilter cannot work with empty training data")
    val trainingSummary = computeFeatureStats(trainData, rawFeatures) // TODO also response summaries??
    log.info("Computed summary stats for training features")
    log.debug(trainingSummary.predictorDistributions.mkString("\n"))

    val scoreData = scoreReader.flatMap{ s =>
      val sd = s.generateDataFrame(rawFeatures, parameters.switchReaderParams()).persist()
      log.info("Loaded scoring data")
      if (sd.count() > 0) Some(sd)
      else {
        log.warn("Scoring dataset was empty. Only training data checks will be used.")
        None
      }
    }

    val scoringSummary = scoreData.map{ sd =>
      val ss = computeFeatureStats(sd, rawFeatures, Some(trainingSummary)) // TODO also response summaries??
      log.info("Computed summary stats for scoring features")
      log.debug(ss.predictorDistributions.mkString("\n"))
      ss
    }

    val (featuresToDropNames, mapKeysToDrop) = getFeaturesToExclude(
      trainingSummary.predictorDistributions.filterNot { case ((name, _), _) => protectedFeatures.contains(name) },
      scoringSummary.toSeq
        .flatMap(_.predictorDistributions.filterNot { case ((name, _), _) => protectedFeatures.contains(name) })
        .toMap,
      trainingSummary.correlationInfo)
    val (featuresToDrop, featuresToKeep) = rawFeatures.partition(rf => featuresToDropNames.contains(rf.name))
    val featuresToKeepNames = Array(DataFrameFieldNames.KeyFieldName) ++ featuresToKeep.map(_.name)

    val featuresDropped = trainData.drop(featuresToDropNames: _*)
    val mapsCleaned = featuresDropped.rdd.map{ row =>
      val kept = featuresToKeepNames.map{ fn =>
        if (mapKeysToDrop.contains(fn)) {
          val map = row.getMapAny(fn)
          if (map != null) map.filterNot{ case (k, _) => mapKeysToDrop(fn).contains(k) } else map
        } else {
          row.getAny(fn)
        }
      }
      Row.fromSeq(kept)
    }

    val schema = StructType(featuresToKeepNames.map(featuresDropped.schema(_)))
    val cleanedData = spark.createDataFrame(mapsCleaned, schema).persist()
    trainData.unpersist()
    scoreData.map(_.unpersist())

    FilteredRawData(cleanedData, featuresToDrop, mapKeysToDrop)
  }
}

/**
 * case class for the RFF filtered data and features to drop
 * @param cleanedData RFF cleaned data
 * @param featuresToDrop raw features dropped by RFF
 * @param mapKeysToDrop keys in map features dropped by RFF
 */
case class FilteredRawData
(
  cleanedData: DataFrame,
  featuresToDrop: Array[OPFeature],
  mapKeysToDrop: Map[String, Set[String]]
)
