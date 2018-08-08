package com.salesforce.op.filters.distrib

import com.salesforce.op.test.TestCommon
import org.scalactic.TolerantNumerics
import org.scalatest.FlatSpec

import scala.util.Random

class DistributionTest extends FlatSpec with TestCommon {
  implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(0.00001)
}
