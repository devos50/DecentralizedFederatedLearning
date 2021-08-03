package nl.tudelft.dfl.configuration

import nl.tudelft.dfl.dataset.Dataset

data class DatasetIteratorConfiguration(
    var batchSize: Int,
    var distribution: List<Int>,
    var maxTestSamples: Int
) {
    companion object {
        fun defaultConfiguration(dataset: Dataset) : DatasetIteratorConfiguration {
            return DatasetIteratorConfiguration(
                dataset.defaultBatchSize,
                dataset.defaultIteratorDistribution.value.toList(),
                10)
        }
    }

    fun copy(): DatasetIteratorConfiguration {
        return DatasetIteratorConfiguration(batchSize, distribution.toList(), maxTestSamples)
    }
}