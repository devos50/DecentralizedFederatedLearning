package nl.tudelft.dfl.demo

import mu.KotlinLogging
import nl.tudelft.dfl.*
import nl.tudelft.dfl.dataset.CustomDatasetType
import nl.tudelft.dfl.demo.nl.tudelft.dfl.EvaluationProcessor
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import java.io.File

fun main() {
    val logger = KotlinLogging.logger("Application")
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

    val evaluationProcessor = EvaluationProcessor(
        baseDirectory,
        "local",
        ArrayList()
    )
    evaluationProcessor.newSimulation("local run", listOf(mlConfiguration), false)

    network.setListeners(
        ScoreIterationListener(5)
    )

    val iterationsBeforeEvaluation = 5
    var epoch = 0
    var iterations = 0
    var iterationsToEvaluation = 0
    val trainConfiguration = mlConfiguration.trainConfiguration
    epochLoop@ while (true) {
        epoch++
        trainDataSetIterator.reset()
        logger.debug { "Starting epoch: $epoch" }
        val start = System.currentTimeMillis()
        while (true) {
            var endEpoch = false
            try {
                network.fit(trainDataSetIterator.next())
            } catch (e: NoSuchElementException) {
                endEpoch = true
            }
            iterations += 1
            iterationsToEvaluation += 1
            logger.debug { "Iteration: $iterations" }

            if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                iterationsToEvaluation = 0
                val end = System.currentTimeMillis()
                evaluationProcessor.evaluate(
                    testDataSetIterator,
                    network,
                    mapOf(),
                    end - start,
                    iterations,
                    epoch,
                    0,
                    false
                )
            }
            if (iterations >= trainConfiguration.maxIteration.value) {
                break@epochLoop
            }
            if (endEpoch) {
                break
            }
        }
    }
}