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

import com.bigml.histogram.Histogram
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.impl.feature.{Inclusion, NumericBucketizer}
import com.twitter.algebird.Semigroup
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import org.apache.spark.mllib.feature.HashingTF

import scala.util.Random

/**
 * Class containing summary information for a feature
 *
 * @param count total count of feature seen
 * @param nulls number of empties seen in feature
 * @param distribution binned counts of feature values (hashed for strings, evently spaced bins for numerics)
 * @param summaryInfo either min and max number of tokens for text data,
 *                    or number of splits used for bins for numeric data
 */
case class FeatureDistribution private[filters] (
    count: Double,
    nullCount: Double,
    private[filters] val histogram: Histogram[Nothing]) {

  final def merge(dist: FeatureDistribution): FeatureDistribution =
    FeatureDistribution(
      count = count + dist.count,
      nullCount = nullCount + dist.nullCount,
      histogram = histogram.merge(dist.histogram).asInstanceOf[Histogram[Nothing]])

  final def update(values: Double*): FeatureDistribution =
    copy(histogram = values.foldLeft(histogram) { case (hist, value) => hist.insert(value) })

  final def cdf(x: Double): Double = histogram.sum(x) / histogram.getTotalCount

  final def density(x: Double): Double = histogram.density(x)

  final def fillRate(): Double = if (count == 0.0) 0.0 else (count - nullCount) / count

  final def maximum(): Double = histogram.getMaximum

  final def minimum(): Double = histogram.getMinimum

  final def relativeFillRate(dist: FeatureDistribution): Double =
    math.abs(fillRate - dist.fillRate)

  final def relativeFillRatio(dist: FeatureDistribution): Double = {
    val (thisFill, thatFill) = (this.fillRate, dist.fillRate)
    val (small, large) = if (thisFill < thatFill) (thisFill, thatFill) else (thatFill, thisFill)

    if (small == 0.0) Double.PositiveInfinity else large / small
  }

  final def cjsDivergence(
    dist: FeatureDistribution,
    n: Int = FeatureDistribution.DefaultMCIntegrationSize): Double =
    jsFunc(dist, d => d.cdf(_), n)

  final def jsDivergence(
    dist: FeatureDistribution,
    n: Int = FeatureDistribution.DefaultMCIntegrationSize): Double =
    jsFunc(dist, d => d.density(_), n)

  private def jsFunc(dist: FeatureDistribution, f: FeatureDistribution => Double => Double, n: Int): Double = {
    val a = math.min(this.minimum, dist.minimum)
    val b = math.max(this.maximum, dist.maximum)
    val (thisF, thatF) = (f(this), f(dist))

    (0.5 / n) * (0 until n).map { _ =>
      val point = (b - a) * Random.nextDouble + a
      val (thisVal, thatVal) = (thisF(point), thatF(point))
      val mainSum =
        if (thisVal == 0.0 || thatVal == 0.0) 0.0 else thisVal * log2(thisVal) + thatVal * log2(thatVal)
      val fSum = thisVal + thatVal

      mainSum - fSum * (log2(0.5) + log2(fSum))
    }.sum
  }

  private def log2(x: Double): Double = math.log(x) / math.log(2)
}

private[op] object FeatureDistribution {

  val DefaultMCIntegrationSize = 1000
  val DefaultMaxBins = 5000

  implicit val semigroup: Semigroup[FeatureDistribution] = new Semigroup[FeatureDistribution] {
    override def plus(l: FeatureDistribution, r: FeatureDistribution) = l.merge(r)
  }

  /**
   * Facilitates feature distribution retrieval from computed feature summaries
   *
   * @param featureKey feature key
   * @param summary feature summary
   * @param value optional processed sequence
   * @param bins number of histogram bins
   * @param hasher hashing method to use for text and categorical features
   * @return feature distribution given the provided information
   */
  def apply(
    value: ProcessedSeq,
    hasher: HashingTF,
    maxBins: Int): FeatureDistribution =
    value match {
      case Left(seq) =>
        FeatureDistribution(maxBins).copy(count = 1).update(hasher.transform(seq).toArray: _*)
      case Right(seq) =>
        FeatureDistribution(maxBins).copy(count = 1).update(seq: _*)
    }

  def apply(maxBins: Int): FeatureDistribution =
    FeatureDistribution(
      count = 0,
      nullCount = 0,
      histogram = new Histogram[Nothing](maxBins))
}
