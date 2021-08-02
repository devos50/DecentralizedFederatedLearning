package nl.tudelft.dfl.attack

import mu.KotlinLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import kotlin.random.Random

private val logger = KotlinLogging.logger("Noise")

class Noise : PoisoningAttack() {
    override fun generateAttack(
        numAttackers: Int,
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: Map<Int, INDArray>,
        random: Random
    ): Map<Int, INDArray> {
        logger.debug { formatName("Noise") }
        val numColumns = oldModel.columns()
        val halfNumColumns = numColumns / 2
        val newModels =
            Array<INDArray>(numAttackers) { NDArray(Array(oldModel.rows()) { FloatArray(numColumns) { random.nextFloat() * (if (it < halfNumColumns) -0.5f else 0.5f) } }) }
        return transformToResult(newModels)
    }
}