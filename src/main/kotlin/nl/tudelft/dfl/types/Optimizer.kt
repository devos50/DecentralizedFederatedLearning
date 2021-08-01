package nl.tudelft.dfl.types

import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.IUpdater
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.learning.config.AMSGrad

enum class Optimizer(
    val id: String,
    val text: String,
    val inst: (LearningRate) -> IUpdater,
) {
    NESTEROVS("nesterovs", "Nesterovs", { learningRate -> Nesterovs(learningRate.schedule) }),
    ADAM("adam", "Adam", { learningRate -> Adam(learningRate.schedule) }),
    SGD("sgd", "SGD", { learningRate -> Sgd(learningRate.schedule) }),
    RMSPROP("rmsprop", "RMSprop", { learningRate -> RmsProp(learningRate.schedule) }),
    AMSGRAD("amsgrad", "AMSGRAD", { learningRate -> AMSGrad(learningRate.schedule) });

    companion object {
        fun load(id: String): Optimizer {
            return values().firstOrNull { it.id == id } ?: throw Exception("Optimizer ${id} not found")
        }
    }
}