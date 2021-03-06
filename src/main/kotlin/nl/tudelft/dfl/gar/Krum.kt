package nl.tudelft.dfl.gar

import mu.KotlinLogging
import nl.tudelft.dfl.dataset.CustomDatasetIterator
import nl.tudelft.dfl.d
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

private val logger = KotlinLogging.logger("Krum")

fun getKrum(models: Array<INDArray>, b: Int): Int {
    val distances = Array(models.size) { DoubleArray(models.size) }
    for (i in models.indices) {
        distances[i][i] = 9999999.0
        for (j in i + 1 until models.size) {
            val distance = models[i].distance2(models[j])
            distances[i][j] = distance
            distances[j][i] = distance
        }
    }
    val summedDistances = distances.map {
        val sorted = it.sorted()
        sorted.take(models.size - b - 2 - 1).sum()  // The additional -1 is because a peer is not a neighbor of itself
    }.toTypedArray()
    return summedDistances.indexOf(summedDistances.minOrNull()!!)
}

class Krum(private val b: Int) : AggregationRule() {
    override fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDatasetIterator,
    ): INDArray {
        logger.debug { formatName("Krum") }
        val modelMap = HashMap<Int, INDArray>()
        val newModel = oldModel.sub(gradient)
        modelMap[-1] = newModel
        modelMap.putAll(newOtherModels)
        val models = modelMap.values.toTypedArray()
        return if (models.size <= b + 2 + 1) {  // The additional +1 is because we need to add the current peer itself
            logger.debug { "Not using KRUM rule because not enough models found..." }
            newModel
        } else {
            val bestCandidate = getKrum(models, b)
            newModel.addi(models[bestCandidate]).divi(2)
        }
    }
}