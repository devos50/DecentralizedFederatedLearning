package nl.tudelft.dfl.gar

import mu.KotlinLogging
import nl.tudelft.dfl.dataset.CustomDatasetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray


abstract class AggregationRule {
    private val mpl = KotlinLogging.logger("AggregationRule")

    abstract fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDatasetIterator,
        countPerPeer: Map<Int, Int>,
        logging: Boolean,
    ): INDArray

    abstract fun isDirectIntegration(): Boolean

    protected fun formatName(name: String): String {
        return "<====      $name      ====>"
    }
}
