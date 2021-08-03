package nl.tudelft.dfl.gar

import org.junit.jupiter.api.Test
import org.nd4j.linalg.factory.Nd4j
import kotlin.test.assertEquals

class TestBridge: TestAggregationRule() {

    @Test
    fun testBridgeThreeModels() {
        val rule = Bridge(1)

        val oldModel = Nd4j.createFromArray(6.0f, 3.0f, 9.0f, 4.0f)
        val gradient = Nd4j.createFromArray(0.0f, 0.0f, 0.0f, 0.0f)
        val receivedModel1 = Nd4j.createFromArray(2.0f, 0.0f, 6.0f, 4.0f)
        val receivedModel2 = Nd4j.createFromArray(0.0f, 1.0f, 2.0f, 3.0f)

        val updatedModel = rule.integrateParameters(network, oldModel, gradient, mapOf(1 to receivedModel1, 2 to receivedModel2), ArrayDeque(), iterator)
        assertEquals(Nd4j.createFromArray(2.0f, 1.0f, 6.0f, 4.0f), updatedModel)
    }

    @Test
    fun testBridgeFiveModels() {
        val rule = Bridge(2)

        val oldModel = Nd4j.createFromArray(6.0f, 3.0f, 9.0f, 4.0f)
        val gradient = Nd4j.createFromArray(0.0f, 0.0f, 0.0f, 0.0f)
        val receivedModel1 = Nd4j.createFromArray(2.0f, 0.0f, 6.0f, 4.0f)
        val receivedModel2 = Nd4j.createFromArray(8.0f, 8.0f, 4.0f, 3.0f)
        val receivedModel3 = Nd4j.createFromArray(7.0f, 9.0f, 11.0f, 8.0f)
        val receivedModel4 = Nd4j.createFromArray(3.0f, 1.0f, 2.0f, 9.0f)

        val updatedModel = rule.integrateParameters(network, oldModel, gradient, mapOf(1 to receivedModel1, 2 to receivedModel2, 3 to receivedModel3, 4 to receivedModel4), ArrayDeque(), iterator)
        assertEquals(Nd4j.createFromArray(6.0f, 3.0f, 6.0f, 4.0f), updatedModel)
    }
}