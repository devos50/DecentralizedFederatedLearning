package nl.tudelft.dfl.configuration

import nl.tudelft.dfl.dataset.Dataset

data class RunConfiguration(
    var dataset: Dataset,
    var numNodes: Int,
    val datasetIteratorConfiguration: DatasetIteratorConfiguration,
    val nnConfiguration: NNConfiguration,
    val trainConfiguration: TrainConfiguration,
    val attackConfiguration: AttackConfiguration) {

    companion object {
        fun defaultConfiguration(dataset: Dataset) : RunConfiguration {
            return RunConfiguration(
                dataset,
                1,
                DatasetIteratorConfiguration.defaultConfiguration(dataset),
                NNConfiguration.defaultConfiguration(dataset),
                TrainConfiguration.defaultConfiguration(),
                AttackConfiguration.defaultConfiguration())
        }
    }
}