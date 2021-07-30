package nl.tudelft.dfl.dataset.mnist

import nl.tudelft.dfl.Behaviors
import nl.tudelft.dfl.dataset.CustomDatasetType
import nl.tudelft.dfl.DatasetIteratorConfiguration
import nl.tudelft.dfl.dataset.CustomBaseDatasetIterator
import nl.tudelft.dfl.dataset.CustomDatasetIterator
import java.io.File


class CustomMnistDataSetIterator(
    val iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: CustomDatasetType,
    behavior: Behaviors,
    transfer: Boolean,
) : CustomBaseDatasetIterator(
    iteratorConfiguration.batchSize.value,
    -1,
    CustomMnistDataFetcher(
        iteratorConfiguration.distribution.toIntArray(),
        seed,
        dataSetType,
        if (dataSetType == CustomDatasetType.TRAIN) Integer.MAX_VALUE else iteratorConfiguration.maxTestSamples.value,
        behavior,
        transfer
    )
), CustomDatasetIterator {
    override val testBatches by lazy { customFetcher.testBatches }

    override fun getLabels(): List<String> {
        return iteratorConfiguration.distribution
            .zip(iteratorConfiguration.distribution.indices)
            .filter { (numSamples, _) -> numSamples > 0 }
            .map { it.second.toString() }
    }

    companion object {
        fun create(
            iteratorConfiguration: DatasetIteratorConfiguration,
            seed: Long,
            dataSetType: CustomDatasetType,
            baseDirectory: File,
            behavior: Behaviors,
            transfer: Boolean
        ): CustomMnistDataSetIterator {
            return CustomMnistDataSetIterator(iteratorConfiguration, seed, dataSetType, behavior, transfer)
        }
    }
}
