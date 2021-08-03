package nl.tudelft.dfl.configuration

import nl.tudelft.dfl.dataset.Dataset
import nl.tudelft.dfl.types.L2Regularization
import nl.tudelft.dfl.types.LearningRate
import nl.tudelft.dfl.types.Momentum
import nl.tudelft.dfl.types.Optimizer


enum class NNConfigurationMode {
    REGULAR, TRANSFER, FROZEN
}

data class NNConfiguration(
    var optimizer: Optimizer,
    var learningRate: LearningRate,
    var momentum: Momentum,
    var l2: L2Regularization
) {
    companion object {
        fun defaultConfiguration(dataset: Dataset) : NNConfiguration {
            return NNConfiguration(
                dataset.defaultOptimizer,
                dataset.defaultLearningRate,
                dataset.defaultMomentum,
                dataset.defaultL2)
        }
    }

    fun copy(): NNConfiguration {
        return NNConfiguration(optimizer, learningRate, momentum, l2)
    }
}