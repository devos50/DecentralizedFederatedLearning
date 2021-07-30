package nl.tudelft.dfl.demo

import nl.tudelft.dfl.*
import nl.tudelft.dfl.dataset.CustomDatasetType
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import java.io.File

fun main() {
    val seed = 1337
    val dataset = Datasets.MNIST
    val baseDirectory = File(".")
    val mlConfiguration = MLConfiguration(
        dataset,
        DatasetIteratorConfiguration(
            batchSize = dataset.defaultBatchSize,
            maxTestSamples = MaxTestSamples.NUM_20,
            distribution = dataset.defaultIteratorDistribution.value.toList()
        ),
        NNConfiguration(
            optimizer = dataset.defaultOptimizer,
            learningRate = dataset.defaultLearningRate,
            momentum = dataset.defaultMomentum,
            l2 = dataset.defaultL2
        ),
        TrainConfiguration(
            maxIteration = MaxIterations.ITER_250,
            gar = GARs.AVERAGE,
            communicationPattern = CommunicationPatterns.RANDOM,
            behavior = Behaviors.BENIGN,
            slowdown = Slowdowns.NONE,
            joiningLate = TransmissionRounds.N0,
            iterationsBeforeEvaluation = 10,
            iterationsBeforeSending = 1,
            transfer = true,
            connectionRatio = 1.0,
            latency = 0
        ),
        ModelPoisoningConfiguration(
            attack = ModelPoisoningAttacks.NONE,
            numAttackers = NumAttackers.NUM_0
        )
    )

    val trainDataSetIterator = mlConfiguration.dataset.inst(
        mlConfiguration.datasetIteratorConfiguration,
        seed.toLong(),
        CustomDatasetType.TRAIN,
        baseDirectory,
        Behaviors.BENIGN,
        false,
    )
    val testDataSetIterator = mlConfiguration.dataset.inst(
        mlConfiguration.datasetIteratorConfiguration,
        seed.toLong() + 1,
        CustomDatasetType.FULL_TEST,
        baseDirectory,
        Behaviors.BENIGN,
        false,
    )

    val network = MultiLayerNetwork(mlConfiguration.dataset.architecture(mlConfiguration.nnConfiguration, seed, NNConfigurationMode.REGULAR))
    network.init()

//    val evaluationProcessor = EvaluationProcessor(
//        baseDirectory,
//        "local",
//        ArrayList()
//    )
//    evaluationProcessor.newSimulation("local run", listOf(mlConfiguration), false)
//    network.setListeners(
//        ScoreIterationListener(printScoreIterations)
//    )
}