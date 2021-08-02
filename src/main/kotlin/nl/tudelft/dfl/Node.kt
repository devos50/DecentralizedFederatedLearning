package nl.tudelft.dfl

import mu.KotlinLogging
import nl.tudelft.dfl.configuration.*
import nl.tudelft.dfl.dataset.CustomDatasetIterator
import nl.tudelft.dfl.dataset.CustomDatasetType
import nl.tudelft.dfl.types.Behavior
import nl.tudelft.dfl.types.CommunicationPattern
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.string.NDArrayStrings
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import kotlin.random.Random

private val logger = KotlinLogging.logger("Node")
private const val SIZE_RECENT_OTHER_MODELS = 20
private const val TEST_SET_SIZE = 10

class Node(
    private val nodeIndex: Int,
    runConfiguration: RunConfiguration,
    baseDirectory: File,
    private val evaluationProcessor: EvaluationProcessor,
    private val start: Long,
) {
    val configuration: RunConfiguration
    var neighbours: List<Node> = listOf()
    private val recentOtherModelsBuffer = ArrayDeque<Pair<Int, INDArray>>()
    private val newOtherModelBufferTemp = Array<MutableMap<Int, INDArray>>(1) { ConcurrentHashMap() }
    private val newOtherModelBuffer = ConcurrentHashMap<Int, INDArray>()
    private val random = Random(nodeIndex)

    var neuralNetwork: MultiLayerNetwork

    var oldParams: INDArray
    var newParams: INDArray
    var gradient: INDArray
    private val iterTrain: CustomDatasetIterator
    private val iterTest: CustomDatasetIterator
    private val iterTestFull: CustomDatasetIterator
    private val logging: Boolean

    init {
        configuration = runConfiguration

        neuralNetwork = if (configuration.trainConfiguration.transfer) {
            loadFromTransferNetwork(File(baseDirectory, "transfer-${configuration.dataset.id}"), configuration.dataset.architecture)
        } else {
            generateNeuralNetwork(configuration.dataset.architecture, nodeIndex, NNConfigurationMode.REGULAR)
        }
        neuralNetwork.outputLayer.params().muli(0)

        oldParams = neuralNetwork.params().dup()
        newParams = NDArray()
        gradient = NDArray()
        val iters = getDataSetIterators(
            configuration.dataset.inst,
            baseDirectory,
        )
        iterTrain = iters[0]
        iterTest = iters[1]
        iterTestFull = iters[2]

        logging = false
    }

    fun generateNeuralNetwork(
        architecture: (nnConfiguration: NNConfiguration, seed: Int, mode: NNConfigurationMode) -> MultiLayerConfiguration,
        seed: Int,
        mode: NNConfigurationMode,
    ): MultiLayerNetwork {
        val network = MultiLayerNetwork(architecture(configuration.nnConfiguration, seed, mode))
        network.init()
        return network
    }

    private fun loadFromTransferNetwork(transferFile: File, generateArchitecture: (nnConfiguration: NNConfiguration, seed: Int, mode: NNConfigurationMode) -> MultiLayerConfiguration): MultiLayerNetwork {
        val transferNetwork = ModelSerializer.restoreMultiLayerNetwork(transferFile)
        val frozenNetwork = generateNeuralNetwork(generateArchitecture, nodeIndex, NNConfigurationMode.FROZEN)
        for ((k, v) in transferNetwork.paramTable()) {
            if (k.split("_")[0].toInt() < transferNetwork.layers.size - 1) {
                frozenNetwork.setParam(k, v.dup())
            }
        }
        return frozenNetwork
    }

    protected fun getDataSetIterators(
        inst: (iteratorConfiguration: DatasetIteratorConfiguration, seed: Long, dataSetType: CustomDatasetType, baseDirectory: File, behavior: Behavior, transfer: Boolean) -> CustomDatasetIterator,
        baseDirectory: File,
    ): List<CustomDatasetIterator> {
        val seed = nodeIndex.toLong() * 10
        val trainDataSetIterator = inst(
            DatasetIteratorConfiguration(
                configuration.datasetIteratorConfiguration.batchSize,
                configuration.datasetIteratorConfiguration.distribution,
                configuration.datasetIteratorConfiguration.maxTestSamples
            ),
            seed,
            CustomDatasetType.TRAIN,
            baseDirectory,
            configuration.trainConfiguration.behavior,
            false,
        )
        logger.debug { "Loaded trainDataSetIterator" }
        val testDataSetIterator = inst(
            DatasetIteratorConfiguration(
                200,
                configuration.datasetIteratorConfiguration.distribution.map { if (it == 0) 0 else TEST_SET_SIZE },
                configuration.datasetIteratorConfiguration.maxTestSamples
            ),
            seed + 1,
            CustomDatasetType.TEST,
            baseDirectory,
            configuration.trainConfiguration.behavior,
            false,
        )
        logger.debug { "Loaded testDataSetIterator" }
        val fullTestDataSetIterator = inst(
            DatasetIteratorConfiguration(
                200,
                List(configuration.datasetIteratorConfiguration.distribution.size) { configuration.datasetIteratorConfiguration.maxTestSamples },
                configuration.datasetIteratorConfiguration.maxTestSamples
            ),
            seed + 2,
            CustomDatasetType.FULL_TEST,
            baseDirectory,
            Behavior.BENIGN,
            false,
        )
        logger.debug { "Loaded fullTestDataSetIterator" }
        return listOf(trainDataSetIterator, testDataSetIterator, fullTestDataSetIterator)
    }

    fun performIteration(epoch: Int, iteration: Int): Boolean {
        newParams = neuralNetwork.params().dup()
        gradient = oldParams.sub(newParams)

        val formatter = NDArrayStrings()
        logger.t(logging) { "5 - outputlayer $nodeIndex: ${formatter.format(neuralNetwork.outputLayer.paramTable().getValue("W"))}" }
        if (configuration.trainConfiguration.behavior == Behavior.BENIGN) {
            if (iteration % configuration.trainConfiguration.iterationsBeforeSending == 0) {
                addPotentialAttacks()
            }
            val start = System.currentTimeMillis()
            potentiallyIntegrateParameters()
            if (iteration < 4) {
                logger.debug { "Measured time for ${configuration.trainConfiguration.gar.text} iteration: ${System.currentTimeMillis() - start}" }
            }
        }
        newOtherModelBuffer.clear()

        logger.t(logging) { "4 - outputlayer $nodeIndex: ${formatter.format(neuralNetwork.outputLayer.paramTable().getValue("W"))}" }

        logger.t(logging) { "3 - outputlayer $nodeIndex: ${formatter.format(neuralNetwork.outputLayer.paramTable().getValue("W"))}" }
        oldParams = neuralNetwork.params().dup()

        val epochEnd = fitNetwork(neuralNetwork, iterTrain)

        if (iteration % configuration.trainConfiguration.iterationsBeforeSending == 0) {
            shareModel(neuralNetwork.params().dup())
        }

        potentiallyEvaluate(epoch, iteration)
        return epochEnd
    }

    private fun shareModel(
        params: INDArray,
    ) {
        val message = craftMessage(params)
        when (configuration.trainConfiguration.communicationPattern) {
            CommunicationPattern.ALL -> neighbours.forEach { it.addNetworkMessage(nodeIndex, message) }
            CommunicationPattern.RANDOM -> neighbours
                .filter { it.nodeIndex != nodeIndex }
                .random().addNetworkMessage(nodeIndex, message)
        }
    }

    protected fun craftMessage(first: INDArray): INDArray {
        return when (configuration.trainConfiguration.behavior) {
            Behavior.BENIGN -> first
            Behavior.NOISE -> craftNoiseMessage(first, random)
            Behavior.LABEL_FLIP_2 -> first
            Behavior.LABEL_FLIP_ALL -> first
        }
    }

    private fun craftNoiseMessage(first: INDArray, random: Random): INDArray {
        val numColumns = first.columns()
        return NDArray(Array(first.rows()) { FloatArray(numColumns) { random.nextFloat() / 2 - 0.2f } })
    }

    private fun fitNetwork(network: MultiLayerNetwork, dataSetIterator: CustomDatasetIterator): Boolean {
        try {
            val ds = dataSetIterator.next()
            network.fit(ds)
        } catch (e: Exception) {
            dataSetIterator.reset()
            return true
        }
        return false
    }

    private fun addPotentialAttacks() {
        val attackVectors = configuration.attackConfiguration.attack.obj.generateAttack(
            configuration.attackConfiguration.numAttackers,
            oldParams,
            gradient,
            newOtherModelBuffer,
            random
        )
        newOtherModelBuffer.putAll(attackVectors)
    }

    private fun potentiallyIntegrateParameters() {
        val numPeers = newOtherModelBuffer.size + 1
        if (numPeers > 1) {
            val averageParams = configuration.trainConfiguration.gar.obj.integrateParameters(
                neuralNetwork,
                oldParams,
                gradient,
                newOtherModelBuffer,
                recentOtherModelsBuffer,
                iterTest,
                logging
            )
            neuralNetwork.setParameters(averageParams)
            recentOtherModelsBuffer.addAll(newOtherModelBuffer.toList())
            while (recentOtherModelsBuffer.size > SIZE_RECENT_OTHER_MODELS) {
                recentOtherModelsBuffer.removeFirst()
            }
        }
    }

    private fun potentiallyEvaluate(epoch: Int, iteration: Int) {
        if (iteration < 20 || iteration % configuration.trainConfiguration.iterationsBeforeEvaluation == 0) {
            val evaluationScript = {
                val elapsedTime2 = System.currentTimeMillis() - start
                evaluationProcessor.evaluate(
                    iterTestFull,
                    neuralNetwork,
                    elapsedTime2,
                    iteration,
                    epoch,
                    nodeIndex,
                    logging
                )
            }
            evaluationScript.invoke()
        }
    }

    fun applyNetworkBuffers() {
        newOtherModelBuffer.putAll(newOtherModelBufferTemp.first())
        (0 until newOtherModelBufferTemp.size - 1).forEach { index -> newOtherModelBufferTemp[index] = newOtherModelBufferTemp[index+1] }
    }

    fun addNetworkMessage(from: Int, message: INDArray) {
        newOtherModelBufferTemp.last()[from] = message.dup()
    }
}
