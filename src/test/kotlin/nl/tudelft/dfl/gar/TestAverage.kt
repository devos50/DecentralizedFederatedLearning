package nl.tudelft.dfl.gar

import org.nd4j.linalg.factory.Nd4j
import kotlin.test.Test
import kotlin.test.assertEquals


class TestAverage: TestAggregationRule() {

    @Test
    fun testAverage() {
        val rule = Average()

        val oldModel = Nd4j.createFromArray(1, 5, 10)
        val gradient = Nd4j.createFromArray(0, 0, 0)
        val newModel = Nd4j.createFromArray(3, 9, 2)
        val updatedModel = rule.integrateParameters(network, oldModel, gradient, mapOf(1 to newModel), ArrayDeque(), iterator)
        assertEquals(Nd4j.createFromArray(2, 7, 6), updatedModel)
    }
}