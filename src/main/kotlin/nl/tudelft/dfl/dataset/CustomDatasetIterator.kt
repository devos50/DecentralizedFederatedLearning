package nl.tudelft.dfl.dataset

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

interface CustomDatasetIterator : DataSetIterator {
    val testBatches: Array<DataSet?>
}
