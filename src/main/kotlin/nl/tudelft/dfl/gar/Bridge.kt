package nl.tudelft.dfl.gar

import mu.KotlinLogging
import nl.tudelft.dfl.dataset.CustomDatasetIterator
import nl.tudelft.dfl.d
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.factory.Nd4j

private val logger = KotlinLogging.logger("Bridge")

fun trimmedMean(b: Int, l: FloatArray): Float {
    l.sort()
    return l.copyOfRange(b, l.size - b).average().toFloat()
}

class Bridge(private val b: Int) : AggregationRule() {
    private val minimumModels = 2 * b + 1

    override fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDatasetIterator,
    ): INDArray {
        logger.debug { formatName("BRIDGE") }
        val models = HashMap<Int, INDArray>()
        val newModel = oldModel.sub(gradient)
        models[-1] = newModel
        models.putAll(newOtherModels)
        return if (models.size < minimumModels) {
            newModel
        } else {
            val modelsAsArrays = models.map { toFloatArray(it.value) }.toTypedArray()
            val newVector = FloatArray(modelsAsArrays[0].size)
            for (i in modelsAsArrays[0].indices) {
                val elements = FloatArray(modelsAsArrays.size)
                modelsAsArrays.forEachIndexed { j, modelsAsArray -> elements[j] = modelsAsArray[i] }
                newVector[i] = trimmedMean(b, elements)
            }
            Nd4j.createFromArray(*newVector)
        }
    }
}