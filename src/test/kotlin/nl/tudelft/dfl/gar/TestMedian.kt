package nl.tudelft.dfl.gar

import org.junit.jupiter.api.Test
import org.nd4j.linalg.factory.Nd4j
import kotlin.test.assertEquals

class TestMedian: TestAggregationRule() {

    @Test
    fun testMedian() {
        val rule = Median()

        val oldModel = Nd4j.createFromArray(1.0f, 5.0f, 10.0f)
        val gradient = Nd4j.createFromArray(0.0f, 0.0f, 0.0f)
        val newModel = Nd4j.createFromArray(4.0f, 10.0f, 2.0f)
        val updatedModel = rule.integrateParameters(network, oldModel, gradient, mapOf(1 to newModel), ArrayDeque(), iterator)
        assertEquals(Nd4j.createFromArray(2.5f, 7.5f, 6.0f), updatedModel)
    }
}