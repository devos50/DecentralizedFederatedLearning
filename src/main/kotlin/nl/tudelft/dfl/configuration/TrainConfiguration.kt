package nl.tudelft.dfl.configuration

import nl.tudelft.dfl.types.Behavior
import nl.tudelft.dfl.types.CommunicationPattern
import nl.tudelft.dfl.types.GAR

data class TrainConfiguration(
    var maxIterations: Int,
    var gar: GAR,
    var communicationPattern: CommunicationPattern,
    var behavior: Behavior,
    var iterationsBeforeEvaluation: Int,
    var iterationsBeforeSending: Int,
    var transfer: Boolean,
) {
    companion object {
        fun defaultConfiguration() : TrainConfiguration {
            return TrainConfiguration(
                200,
                GAR.AVERAGE,
                CommunicationPattern.ALL,
                Behavior.BENIGN,
                10,
                1,
                false)
        }
    }
}