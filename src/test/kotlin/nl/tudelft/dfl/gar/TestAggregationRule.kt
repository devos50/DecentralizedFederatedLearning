package nl.tudelft.dfl.gar

import nl.tudelft.dfl.configuration.DatasetIteratorConfiguration
import nl.tudelft.dfl.configuration.NNConfiguration
import nl.tudelft.dfl.configuration.NNConfigurationMode
import nl.tudelft.dfl.dataset.CustomDatasetIterator
import nl.tudelft.dfl.dataset.CustomDatasetType
import nl.tudelft.dfl.dataset.Dataset
import nl.tudelft.dfl.types.Behavior
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import java.io.File

abstract class TestAggregationRule {

    val network: MultiLayerNetwork
    val iterator: CustomDatasetIterator

    init {
        network = MultiLayerNetwork(Dataset.MNIST.architecture(NNConfiguration.defaultConfiguration(Dataset.MNIST), 1, NNConfigurationMode.REGULAR))
        iterator = Dataset.MNIST.inst(DatasetIteratorConfiguration.defaultConfiguration(Dataset.MNIST), 1, CustomDatasetType.TEST, File("."), Behavior.BENIGN, false)
    }

}