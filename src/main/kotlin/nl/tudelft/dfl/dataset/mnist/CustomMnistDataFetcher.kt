package nl.tudelft.dfl.dataset.mnist

import mu.KotlinLogging
import nl.tudelft.dfl.dataset.CustomDatasetType
import nl.tudelft.dfl.dataset.CustomBaseDataFetcher
import nl.tudelft.dfl.types.Behavior
import org.apache.commons.io.FilenameUtils
import org.apache.commons.io.FileUtils
import org.deeplearning4j.common.resources.DL4JResources
import org.deeplearning4j.common.resources.ResourceType
import org.deeplearning4j.datasets.base.EmnistFetcher
import org.deeplearning4j.datasets.base.MnistFetcher
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.File
import java.util.stream.IntStream

private val logger = KotlinLogging.logger("CustomMnistDataFetcher")
private const val NUM_EMNIST_TRAINING_EXAMPLES = 124800
private const val NUM_EMNIST_TESTING_EXAMPLES = 20800
private const val NUM_EMNIST_CLASSES = 26
private const val NUM_MNIST_CLASSES = 10
private const val NUM_EMNIST_EXAMPLES_PER_CLASS = 2200
private const val SIZE_IMAGE = 28 * 28
private const val FILENAME_EMNIST_TRAIN_IMAGES = "emnist-letters-train-images-idx3-ubyte"
private const val FILENAME_EMNIST_TRAIN_LABELS = "emnist-letters-train-images-idx3-ubyte"
private const val FILENAME_EMNIST_TEST_IMAGES = "emnist-letters-train-images-idx3-ubyte"
private const val FILENAME_EMNIST_TEST_LABELS = "emnist-letters-train-images-idx3-ubyte"

class CustomMnistDataFetcher(
    val iteratorDistribution: IntArray,
    seed: Long,
    val dataSetType: CustomDatasetType,
    maxTestSamples: Int,
    behavior: Behavior,
) : CustomBaseDataFetcher(seed) {
    override val testBatches by lazy { createTestBatches() }

    @Transient
    private var man: CustomMnistManager
    private var featureData = Array(1) { FloatArray(SIZE_IMAGE) }

    init {
        if (!mnistExists()) {
            MnistFetcher().downloadAndUntar()
        }
        val mnistRoot = DL4JResources.getDirectory(ResourceType.DATASET, "MNIST").absolutePath
        val images: String
        val labels: String
        val numExamples: Int
        if (dataSetType == CustomDatasetType.TRAIN) {
            images = FilenameUtils.concat(mnistRoot, MnistFetcher.TRAINING_FILES_FILENAME_UNZIPPED)
            labels = FilenameUtils.concat(mnistRoot, MnistFetcher.TRAINING_FILE_LABELS_FILENAME_UNZIPPED)
            numExamples = MnistDataFetcher.NUM_EXAMPLES
        } else {
            images = FilenameUtils.concat(mnistRoot, MnistFetcher.TEST_FILES_FILENAME_UNZIPPED)
            labels = FilenameUtils.concat(mnistRoot, MnistFetcher.TEST_FILE_LABELS_FILENAME_UNZIPPED)
            numExamples = MnistDataFetcher.NUM_EXAMPLES_TEST
        }
        val createMan = {
            CustomMnistManager(
                images,
                labels,
                numExamples,
                iteratorDistribution,
                maxTestSamples,
                seed,
                if (dataSetType == CustomDatasetType.FULL_TEST) Behavior.BENIGN else behavior
            )
        }
        try {
            man = createMan.invoke()
        } catch (e: Exception) {
            try {
                FileUtils.deleteDirectory(File(mnistRoot))
            } catch (e2: Exception) {
                // Ignore
            }
            MnistFetcher().downloadAndUntar()
            man = createMan.invoke()
        }
        totalExamples = man.getNumSamples()
        numOutcomes = NUM_MNIST_CLASSES
        cursor = 0
        inputColumns = man.getInputColumns()
        order = IntStream.range(0, totalExamples).toArray()
        reset() //Shuffle order
    }

    private fun mnistExists(): Boolean {
        val mnistRoot = DL4JResources.getDirectory(ResourceType.DATASET, "MNIST").absolutePath
        var f = File(mnistRoot, MnistFetcher.TRAINING_FILES_FILENAME_UNZIPPED)
        if (!f.exists()) return false
        f = File(mnistRoot, MnistFetcher.TRAINING_FILE_LABELS_FILENAME_UNZIPPED)
        if (!f.exists()) return false
        f = File(mnistRoot, MnistFetcher.TEST_FILES_FILENAME_UNZIPPED)
        if (!f.exists()) return false
        f = File(mnistRoot, MnistFetcher.TEST_FILE_LABELS_FILENAME_UNZIPPED)
        return f.exists()
    }

    override fun fetch(numExamples: Int) {
        check(hasMore()) { "Unable to get more; there are no more images" }
        var labels = Nd4j.zeros(DataType.FLOAT, numExamples.toLong(), numOutcomes.toLong())
        if (featureData.size != numExamples) {
            featureData = Array(numExamples) { FloatArray(SIZE_IMAGE) }
        }
        var actualExamples = 0
        for (i in 0 until numExamples) {
            if (!hasMore()) break
            val (image, label) = man.readEntry(order[cursor])
            featureData[actualExamples] = image
            labels.put(actualExamples, label, 1.0f)
            actualExamples++
            cursor++
        }
        val features = Nd4j.create(
            if (featureData.size == actualExamples) featureData
            else featureData.copyOfRange(0, actualExamples)
        )
        if (actualExamples < numExamples) {
            labels = labels[NDArrayIndex.interval(0, actualExamples), NDArrayIndex.all()]
        }
        features.divi(255.0)
        curr = DataSet(features, labels)
        if (dataSetType != CustomDatasetType.TRAIN) {
            /**
             * Non-training iterators are called not by the next() function but by supplying the whole iterator
             * => require a reset before getting next batch, otherwise the iterator keeps iterating
             */
            cursor = totalExamples
        }
    }

    private fun createTestBatches(): Array<DataSet?> {
        val testBatches = man.createTestBatches()
        if (featureData.size != TEST_BATCH_SIZE) {
            featureData = Array(TEST_BATCH_SIZE) { FloatArray(SIZE_IMAGE) }
        }
        val result = arrayListOf<DataSet?>()
        for ((label, batch) in testBatches.withIndex()) {
            result.add(
                if (batch.isEmpty()) null
                else createTestBatch(label, batch)
            )
        }
        return result.toTypedArray()
    }

    private fun createTestBatch(label: Int, batch: Array<FloatArray>): DataSet {
        val numSamplesInBatch = batch.size
        val labels = Nd4j.zeros(DataType.FLOAT, numSamplesInBatch.toLong(), numOutcomes.toLong())
        for ((i, img) in batch.withIndex()) {
            labels.put(i, label, 1.0f)
            featureData[i] = img
        }
        val features = Nd4j.create(featureData)
        features.divi(255.0)
        return DataSet(features, labels)
    }

    val labels: List<String>
        get() = man.getLabels()

    companion object {
        const val TEST_BATCH_SIZE = 10
    }
}
