package nl.tudelft.dfl.attack

import org.nd4j.linalg.api.ndarray.INDArray
import kotlin.random.Random

abstract class PoisoningAttack {
    abstract fun generateAttack(
        numAttackers: Int,
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: Map<Int, INDArray>,
        random: Random
    ): Map<Int, INDArray>

    protected fun formatName(name: String): String {
        return "<====      $name      ====>"
    }

    protected fun transformToResult(newModels: Array<INDArray>): Map<Int, INDArray> {
        var attackNum = -1
        return newModels.map {
            attackNum--
            Pair(attackNum, it)
        }.toMap()
    }
}
