package nl.tudelft.dfl.dataset.mnist

import nl.tudelft.dfl.configuration.DatasetIteratorConfiguration
import nl.tudelft.dfl.dataset.CustomDatasetType
import nl.tudelft.dfl.dataset.CustomBaseDatasetIterator
import nl.tudelft.dfl.dataset.CustomDatasetIterator
import nl.tudelft.dfl.types.Behavior
import java.io.File


class CustomMnistDataSetIterator(
    val iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: CustomDatasetType,
    behavior: Behavior,
    transfer: Boolean,
) : CustomBaseDatasetIterator(
    iteratorConfiguration.batchSize,
    -1,
    CustomMnistDataFetcher(
        iteratorConfiguration.distribution.toIntArray(),
        seed,
        dataSetType,
        if (dataSetType == CustomDatasetType.TRAIN) Integer.MAX_VALUE else iteratorConfiguration.maxTestSamples,
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
            behavior: Behavior,
            transfer: Boolean
        ): CustomMnistDataSetIterator {
            return CustomMnistDataSetIterator(iteratorConfiguration, seed, dataSetType, behavior, transfer)
        }
    }
}
