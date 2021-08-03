package nl.tudelft.dfl.gar

import nl.tudelft.dfl.dataset.CustomDatasetIterator
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray


abstract class AggregationRule {

    abstract fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDatasetIterator,
    ): INDArray

    protected fun formatName(name: String): String {
        return "<====      $name      ====>"
    }

    protected fun toFloatArray(first: INDArray): FloatArray {
        val data = first.data()
        val length = data.length().toInt()
        val indexer = data.indexer() as FloatRawIndexer
        val array = FloatArray(length)
        for (i in 0 until length) {
            array[i] = indexer.getRaw(i.toLong())
        }
        return array
    }
}
