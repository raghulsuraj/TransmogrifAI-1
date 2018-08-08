package com.salesforce.op.filters.distrib

import com.salesforce.op.features.types.Binary
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import scala.util.Random

@RunWith(classOf[JUnitRunner])
class MultinomialDistributionTest extends DistributionTest {

  Spec(MultinomialDistribution.getClass) should "work as Binomial with Binary types" in {

    val random = new Random(0L)

    def bernoulliTrial(p: Double): Binary = Binary {
      if (random.nextDouble <= 0.1) None
      else Option(if (random.nextDouble <= p) true else false)
    }

    val p25 = (0 until 1000).map(_ => bernoulliTrial(0.25))
        .foldLeft(MultinomialDistribution.Numeric()) { (dist, binary) => dist.update(binary) }
    val p50 = (0 until 1000).map(_ => bernoulliTrial(0.5))
        .foldLeft(MultinomialDistribution.Numeric()) { (dist, binary) => dist.update(binary) }
    val p75 = (0 until 1000).map(_ => bernoulliTrial(0.75))
        .foldLeft(MultinomialDistribution.Numeric()) { (dist, binary) => dist.update(binary) }

    p25.values should contain theSameElementsAs Seq[Double](0, 1)
    p50.values should contain theSameElementsAs Seq[Double](0, 1)
    p75.values should contain theSameElementsAs Seq[Double](0, 1)

    p25.cdf(0) should equal (0.75 +- 0.025)
    p25.mass(0) should equal (0.75 +- 0.025)
    p25.mass(1) shouldEqual 1 - p25.mass(0)
    p25.cdf(1) shouldEqual 1.0
    p50.cdf(0) should equal (0.5 +- 0.025)
    p50.mass(0) should equal (0.5 +- 0.025)
    p50.mass(1) shouldEqual 1 - p50.mass(0)
    p50.cdf(1) shouldEqual 1.0
    p75.cdf(0) should equal (0.25 +- 0.025)
    p75.mass(0) should equal (0.25 +- 0.025)
    p75.mass(1) shouldEqual 1 - p75.mass(0)
    p75.cdf(1) shouldEqual 1.0
  }
}
