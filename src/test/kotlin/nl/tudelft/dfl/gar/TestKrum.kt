package nl.tudelft.dfl.gar

import org.junit.jupiter.api.Test
import org.nd4j.linalg.factory.Nd4j
import kotlin.test.assertEquals

class TestKrum: TestAggregationRule() {

    @Test
    fun testKrumNotEnoughModels() {
        val rule = Krum(1)

        val oldModel = Nd4j.createFromArray(1.0f, 5.0f, 10.0f)
        val gradient = Nd4j.createFromArray(0.0f, 0.0f, 0.0f)
        val newModel = Nd4j.createFromArray(4.0f, 10.0f, 2.0f)
        val updatedModel = rule.integrateParameters(network, oldModel, gradient, mapOf(1 to newModel), ArrayDeque(), iterator)
        assertEquals(oldModel, updatedModel)
    }

    @Test
    fun testKrum() {
        val rule = Krum(1)

        val oldModel = Nd4j.createFromArray(1.0f, 5.0f, 10.0f)
        val gradient = Nd4j.createFromArray(0.0f, 0.0f, 0.0f)
        val receivedModel = Nd4j.createFromArray(4.0f, 10.0f, 2.0f)
        val updatedModel = rule.integrateParameters(network, oldModel, gradient, mapOf(1 to receivedModel, 2 to receivedModel, 3 to receivedModel, 4 to receivedModel), ArrayDeque(), iterator)
        assertEquals(Nd4j.createFromArray(2.5f, 7.5f, 6.0f), updatedModel)
    }

}