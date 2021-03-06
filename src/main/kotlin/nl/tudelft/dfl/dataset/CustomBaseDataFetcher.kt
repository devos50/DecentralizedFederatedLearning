package nl.tudelft.dfl.dataset

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher
import kotlin.random.Random

abstract class CustomBaseDataFetcher(seed: Long) : BaseDataFetcher() {
    abstract val testBatches: Array<DataSet?>
    protected lateinit var order: IntArray
    protected var rng = Random(seed)

    override fun reset() {
        cursor = 0
        curr = null
        order.shuffle(rng)
    }
}
