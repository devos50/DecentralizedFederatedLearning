package nl.tudelft.dfl.gar

import mu.KotlinLogging
import nl.tudelft.dfl.dataset.CustomDatasetIterator
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

private val logger = KotlinLogging.logger("Median")

fun medianHelper(l: FloatArray): Float {
    l.sort()
    return (l[l.size / 2] + l[(l.size - 1) / 2]) / 2
}

class Median : AggregationRule() {

    override fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDatasetIterator,
    ): INDArray {
        logger.debug { formatName("Median") }
        val models = HashMap<Int, INDArray>()
        models[-1] = oldModel.sub(gradient)
        models.putAll(newOtherModels)
        return median(models)
    }

    private fun median(models: HashMap<Int, INDArray>): INDArray {
        /*
            The "intuitive" way to implement this (see commented code below) is extremely (!) slow due to a horrible
            implementation of the median() function in the dl4j library. It's approximately 130x faster to implement our own median
            function in Kotlin (see uncommented code).

        val result = NDArray(intArrayOf(models.size, models.values.first().shape()[1].toInt()))
        models.onEachIndexed { i, u -> result.putRow(i.toLong(), u.value) }
        return result.median(0).reshape(intArrayOf(1, models.values.first().shape()[1].toInt()))*/

        val modelsAsArrays = models.map { toFloatArray(it.value) }.toTypedArray()
        val newVector = FloatArray(modelsAsArrays[0].size)
        for (i in modelsAsArrays[0].indices) {
            val elements = FloatArray(modelsAsArrays.size)
            modelsAsArrays.forEachIndexed { j, modelsAsArray -> elements[j] = modelsAsArray[i] }
            newVector[i] = medianHelper(elements)
        }
        return Nd4j.createFromArray(*newVector)
    }
}