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

package com.salesforce.op.filters.distrib

import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.salesforce.op.features.types.{FeatureType, OPNumeric, Text}
import com.salesforce.op.stages.impl.feature.{HashAlgorithm, TextTokenizer}
import com.salesforce.op.utils.text.Language
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.{SparseVector, Vector}

abstract class MultinomialDistribution[T <: FeatureType](
    val count: Double,
    val nullCount: Double,
    points: Map[Double, Double])
  extends DiscreteDistribution[T] {

  def extract: T => Map[Double, Double]

  def maximum: Double =
    points.map(_._1).foldLeft(Double.NegativeInfinity)(math.max(_, _))

  def minimum: Double =
    points.map(_._1).foldLeft(Double.PositiveInfinity)(math.min(_, _))

  def values: Set[Double] = points.keySet

  def cdf(x: Double): Double =
    if (count == 0) 0
    else points.collect {
      case (k, v) if k <= x => v
    }.sum / count

  def mass(x: Double): Double =
    if (count <= 0) 0.0 else points.get(x).getOrElse(0.0) / count

  def update(value: T): MultinomialDistribution[T] = {
    val f = extract
    val (newCount, newNullCount, newPoints) =
      if (value.isEmpty) {
        (count, nullCount + 1, points)
      } else {
        (count + 1, nullCount, points.toMap + f(value))
      }

    new MultinomialDistribution[T](newCount, newNullCount, newPoints) {
      val extract: T => Map[Double, Double] = f
    }
  }
}

object MultinomialDistribution {

  object Numeric {
    def apply(
      count: Double = 0,
      nullCount: Double = 0,
      points: Map[Double, Double] = Map()): MultinomialDistribution[OPNumeric[_]] =
      new MultinomialDistribution[OPNumeric[_]](count, nullCount, points) {
        val extract: OPNumeric[_] => Map[Double, Double] = _.toDouble.map(_ -> 1.0).toMap
      }
  }

  object Text {
    val DefaultHasher = tfHasher(1000)

    def apply(
      count: Double = 0,
      nullCount: Double = 0,
      points: Map[Double, Double] = Map(),
      hasher: String => Map[Double, Double] = DefaultHasher): MultinomialDistribution[Text] =
      new MultinomialDistribution[Text](count, nullCount, points) {
        val extract: Text => Map[Double, Double] = _.value.map(hasher).getOrElse(Map())
      }

    def tfHasher(numBins: Int): String => Map[Double, Double] = {
      val tokenize: String => Seq[String] = s => TextTokenizer.Analyzer.analyze(s, Language.Unknown)
      val hasher = new HashingTF(numFeatures = numBins)
        .setBinary(false)
        .setHashAlgorithm(HashAlgorithm.MurMur3.toString.toLowerCase)

      s => hasher.transform(tokenize(s)).compressed match {
        case v: SparseVector =>
          v.indices.map(i => i.toDouble -> v(i)).toMap
        case v: Vector =>
          v.toArray.zipWithIndex.map { case (d, i) => i.toDouble -> d }.toMap
      }
    }
  }
}
