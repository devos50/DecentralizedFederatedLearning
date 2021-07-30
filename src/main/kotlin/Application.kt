package nl.tudelft.dfl.demo

import mu.KotlinLogging
import nl.tudelft.dfl.*
import nl.tudelft.dfl.dataset.CustomDatasetType
import nl.tudelft.dfl.demo.nl.tudelft.dfl.EvaluationProcessor
import nl.tudelft.dfl.demo.nl.tudelft.dfl.Runner
import java.io.File
import kotlin.collections.ArrayList

val logger = KotlinLogging.logger("Application")

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
            maxIteration = MaxIterations.ITER_1000,
            gar = GARs.AVERAGE,
            communicationPattern = CommunicationPatterns.RANDOM,
            behavior = Behaviors.BENIGN,
            slowdown = Slowdowns.NONE,
            joiningLate = TransmissionRounds.N0,
            iterationsBeforeEvaluation = 10,
            iterationsBeforeSending = 1,
            transfer = false,
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

    val evaluationProcessor = EvaluationProcessor(
        baseDirectory,
        "local",
        ArrayList()
    )

    val configs = listOf(mlConfiguration)

    val runner = Runner()
    runner.performTest(baseDirectory, "test", configs, evaluationProcessor)
}