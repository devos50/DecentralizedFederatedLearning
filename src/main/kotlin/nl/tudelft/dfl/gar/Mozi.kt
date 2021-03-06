package nl.tudelft.dfl.gar

import mu.KotlinLogging
import nl.tudelft.dfl.dataset.CustomDatasetIterator
import nl.tudelft.dfl.d
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import kotlin.math.ceil

private val logger = KotlinLogging.logger("Mozi")

class Mozi(private val fracBenign: Double) : AggregationRule() {
    private val TEST_BATCH = 100

    override fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDatasetIterator,
    ): INDArray {
        logger.debug { formatName("MOZI") }
        logger.debug { "Found ${newOtherModels.size} other models" }
        logger.debug { "oldModel: " + oldModel.getDouble(0) }
        val Ndistance = applyDistanceFilter(oldModel, newOtherModels)
        logger.debug { "After distance filter, remaining:${Ndistance.size}" }
        val Nperformance = applyPerformanceFilter(oldModel, Ndistance, network, testDataSetIterator)
        logger.debug { "After performance filter, remaining:${Nperformance.size}" }
//        if (Nperformance.isEmpty()) {
//            logger.debug("Nperformance empty => taking ${Ndistance[0].getDouble(0)}")
//            Nperformance = arrayListOf(Ndistance[0])
//        }

        // This is not included in the original algorithm!!!!
        if (Nperformance.isEmpty()) {
            return oldModel.sub(gradient)
        }

        val Rmozi = average(Nperformance)
        logger.debug("average: ${Rmozi.getDouble(0)}")
        val alpha = 0.5
        val part1 = oldModel.sub(gradient).muli(alpha)
        val result = part1.addi(Rmozi.muli(1 - alpha))
        logger.debug("result: ${result.getDouble(0)}")
        return result
    }

    private fun calculateLoss(
        model: INDArray,
        network: MultiLayerNetwork,
        sample: DataSet
    ): Double {
        network.setParameters(model)
        return network.score(sample)
    }

    private fun calculateLoss(
        models: Array<INDArray>,
        network: MultiLayerNetwork,
        sample: DataSet,
        logging: Boolean
    ): DoubleArray {
        val scores = DoubleArray(models.size)
        for (model in models.withIndex()) {
            network.setParameters(model.value)
            scores[model.index] = network.score(sample)
            logger.debug { "otherLoss = ${scores[model.index]}" }
        }
        return scores
    }

    private fun applyDistanceFilter(
        oldModel: INDArray,
        newOtherModels: Map<Int, INDArray>,
    ): Array<INDArray> {
        val distances = hashMapOf<Int, Double>()
        for (otherModel in newOtherModels) {
            val distance = oldModel.distance2(otherModel.value)
            logger.debug { "Distance calculated: $distance" }
            distances[otherModel.key] = distance
        }
        val sortedDistances = distances.toList().sortedBy { it.second }.toMap()
        val numBenign = ceil(fracBenign * newOtherModels.size).toInt()
        logger.debug { "#benign: $numBenign" }
        return sortedDistances
            .keys
            .take(numBenign)
            .map { newOtherModels.getValue(it) }
            .toTypedArray()
    }

    private fun applyPerformanceFilter(
        oldModel: INDArray,
        newOtherModels: Array<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator,
    ): Array<INDArray> {
        val result = arrayListOf<INDArray>()
        testDataSetIterator.reset()
        val sample = testDataSetIterator.next(TEST_BATCH)
        val oldLoss = calculateLoss(oldModel, network, sample)
        logger.debug { "oldLoss: $oldLoss" }
        val otherLosses = calculateLoss(newOtherModels, network, sample, true)
        for ((index, otherLoss) in otherLosses.withIndex()) {
            logger.debug { "otherLoss $index: $otherLoss" }
            if (otherLoss <= oldLoss) {
                result.add(newOtherModels[index])
                logger.debug { "ADDING model($index): " + newOtherModels[index].getDouble(0) }
            } else {
                logger.debug { "NOT adding model($index): " + newOtherModels[index].getDouble(0) }
            }
        }
        return result.toTypedArray()
    }

    private fun average(list: Array<INDArray>): INDArray {
        var arr: INDArray? = null
        list.forEachIndexed { i, model ->
            if (i == 0) {
                arr = model.dup()
            } else {
                arr!!.addi(model)
            }
        }
        return arr!!.divi(list.size)
    }
}