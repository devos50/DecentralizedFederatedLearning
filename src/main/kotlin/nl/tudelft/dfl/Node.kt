package nl.tudelft.dfl

import mu.KotlinLogging
import nl.tudelft.dfl.dataset.CustomDatasetIterator
import nl.tudelft.dfl.dataset.CustomDatasetType
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.string.NDArrayStrings
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.round
import kotlin.random.Random

private val logger = KotlinLogging.logger("Node")
private const val ONLY_EVALUATE_FIRST_NODE = false
private const val SIZE_RECENT_OTHER_MODELS = 20

class Node(
    private val nodeIndex: Int,
    testConfig: MLConfiguration,
    private val generateNetwork: (architecture: (nnConfiguration: NNConfiguration, seed: Int, mode: NNConfigurationMode) -> MultiLayerConfiguration, nnConfiguration: NNConfiguration, seed: Int, mode: NNConfigurationMode) -> MultiLayerNetwork,
    getDataSetIterators: (inst: (iteratorConfiguration: DatasetIteratorConfiguration, seed: Long, dataSetType: CustomDatasetType, baseDirectory: File, behavior: Behaviors, transfer: Boolean) -> CustomDatasetIterator, datasetIteratorConfiguration: DatasetIteratorConfiguration, seed: Long, baseDirectory: File, behavior: Behaviors) -> List<CustomDatasetIterator>,
    baseDirectory: File,
    private val evaluationProcessor: EvaluationProcessor,
    private val start: Long,
    val shareModel: (params: INDArray, trainConfiguration: TrainConfiguration, random: Random, nodeIndex: Int, countPerPeer: Map<Int, Int>) -> Unit,
) {
    val formatter = NDArrayStrings()
    private val dataset = testConfig.dataset
    private val recentOtherModelsBuffer = ArrayDeque<Pair<Int, INDArray>>()
    private val newOtherModelBufferTemp = Array<MutableMap<Int, INDArray>>(testConfig.trainConfiguration.latency + 1) { ConcurrentHashMap() }
    private val newOtherModelBuffer = ConcurrentHashMap<Int, INDArray>()
    private val random = Random(nodeIndex)

    var network: MultiLayerNetwork
    private val labels: List<String>

    private val datasetIteratorConfiguration: DatasetIteratorConfiguration
    private val distribution: List<Int>
    private val usedClassIndices: List<Int>

    private val nnConfiguration: NNConfiguration

    private val trainConfiguration: TrainConfiguration
    public val behavior: Behaviors
    private val iterationsBeforeEvaluation: Int
    private val iterationsBeforeSending: Int
    private var joiningLateRemainingIterations: Int
    private val slowdown: Slowdowns
    private val gar: GARs
    private val fromTransfer: Boolean

    private val modelPoisoningConfiguration: ModelPoisoningConfiguration
    private val modelPoisoningAttack: ModelPoisoningAttacks
    private val numAttackers: NumAttackers

    var oldParams: INDArray
    var newParams: INDArray
    var gradient: INDArray
    private val iterTrain: CustomDatasetIterator
    private val iterTest: CustomDatasetIterator
    private val iterTestFull: CustomDatasetIterator
    private val logging: Boolean

    private lateinit var cw: INDArray
    private lateinit var countPerPeer: Map<Int, Int>
    private var slowdownRemainingIterations = 0


    init {
        datasetIteratorConfiguration = testConfig.datasetIteratorConfiguration
        distribution = datasetIteratorConfiguration.distribution
        usedClassIndices = distribution.mapIndexed { ind, v -> if (v > 0) ind else null }.filterNotNull()

        nnConfiguration = testConfig.nnConfiguration

        trainConfiguration = testConfig.trainConfiguration
        behavior = trainConfiguration.behavior
        iterationsBeforeEvaluation = trainConfiguration.iterationsBeforeEvaluation
        iterationsBeforeSending = trainConfiguration.iterationsBeforeSending
        joiningLateRemainingIterations = trainConfiguration.joiningLate.rounds * iterationsBeforeSending
        slowdown = trainConfiguration.slowdown
        gar = trainConfiguration.gar
        fromTransfer = trainConfiguration.transfer

        modelPoisoningConfiguration = testConfig.modelPoisoningConfiguration
        modelPoisoningAttack = modelPoisoningConfiguration.attack
        numAttackers = modelPoisoningConfiguration.numAttackers

        network = if (fromTransfer) {
            loadFromTransferNetwork(File(baseDirectory, "transfer-${dataset.id}"), dataset.architecture)
        } else {
            generateNetwork(dataset.architecture, testConfig.nnConfiguration, nodeIndex, NNConfigurationMode.REGULAR)
        }
        network.outputLayer.params().muli(0)

        oldParams = network.params().dup()
        newParams = NDArray()
        gradient = NDArray()
        val iters = getDataSetIterators(
            dataset.inst,
            datasetIteratorConfiguration,
            nodeIndex.toLong() * 10,
            baseDirectory,
            behavior
        )
        iterTrain = iters[0]
        iterTest = iters[1]
        iterTestFull = iters[2]

        labels = iterTrain.labels

        logging = false
    }

    private fun loadFromTransferNetwork(transferFile: File, generateArchitecture: (nnConfiguration: NNConfiguration, seed: Int, mode: NNConfigurationMode) -> MultiLayerConfiguration): MultiLayerNetwork {
        val transferNetwork = ModelSerializer.restoreMultiLayerNetwork(transferFile)
        val frozenNetwork = generateNetwork(generateArchitecture, nnConfiguration, nodeIndex, NNConfigurationMode.FROZEN)
        for ((k, v) in transferNetwork.paramTable()) {
            if (k.split("_")[0].toInt() < transferNetwork.layers.size - 1) {
                frozenNetwork.setParam(k, v.dup())
            }
        }
        return frozenNetwork
    }

    fun performIteration(epoch: Int, iteration: Int): Boolean {
        newParams = network.params().dup()
        gradient = oldParams.sub(newParams)

        if (joiningLateSkip()) {
            return false
        }
        if (slowdownSkip()) {
            return false
        }

        logger.t(logging) { "5 - outputlayer $nodeIndex: ${formatter.format(network.outputLayer.paramTable().getValue("W"))}" }
        if (behavior == Behaviors.BENIGN) {
            if (iteration % iterationsBeforeSending == 0) {
                addPotentialAttacks()
            }
            val start = System.currentTimeMillis()
            potentiallyIntegrateParameters(iteration)
            if (iteration < 4) {
                logger.debug { "Measured time for ${gar.text} iteration: ${System.currentTimeMillis() - start}" }
            }
        }
        newOtherModelBuffer.clear()

        logger.t(logging) { "4 - outputlayer $nodeIndex: ${formatter.format(network.outputLayer.paramTable().getValue("W"))}" }

        logger.t(logging) { "3 - outputlayer $nodeIndex: ${formatter.format(network.outputLayer.paramTable().getValue("W"))}" }
        oldParams = network.params().dup()

        val epochEnd = fitNetwork(network, iterTrain)

        logger.t(logging) { "2 - outputlayer $nodeIndex: ${formatter.format(cw)}" }

        if (iteration % iterationsBeforeSending == 0) {
            logger.t(logging) { "1... - cw $nodeIndex: ${formatter.format(cw) }"}
            shareModel(
                network.params().dup(),
                trainConfiguration,
                random,
                nodeIndex,
                countPerPeer
            )
        }

        potentiallyEvaluate(epoch, iteration)
        return epochEnd
    }

    private fun joiningLateSkip(): Boolean {
        if (joiningLateRemainingIterations > 0) {
            joiningLateRemainingIterations--
            if (nodeIndex == 0) logger.debug { "JL => continue" }
            newOtherModelBuffer.clear()
            return true
        }
        return false
    }

    private fun slowdownSkip(): Boolean {
        if (slowdown != Slowdowns.NONE) {
            if (slowdownRemainingIterations > 0) {
                slowdownRemainingIterations--
                if (nodeIndex == 0) logger.debug { "SD => continue" }
                newOtherModelBuffer.clear()
                return true
            } else {
                slowdownRemainingIterations = round(1 / slowdown.multiplier).toInt() - 1
            }
        }
        return false
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
        val attackVectors = modelPoisoningAttack.obj.generateAttack(
            numAttackers,
            oldParams,
            gradient,
            newOtherModelBuffer,
            random
        )
        newOtherModelBuffer.putAll(attackVectors)
    }

    private fun potentiallyIntegrateParameters(iteration: Int) {
        val numPeers = newOtherModelBuffer.size + 1
        if (numPeers > 1) {
            val averageParams = gar.obj.integrateParameters(
                network,
                oldParams,
                gradient,
                newOtherModelBuffer,
                recentOtherModelsBuffer,
                iterTest,
                countPerPeer,
                logging/* && (iteration % iterationsBeforeEvaluation == 0)*/
            )
            network.setParameters(averageParams)
            recentOtherModelsBuffer.addAll(newOtherModelBuffer.toList())
            while (recentOtherModelsBuffer.size > SIZE_RECENT_OTHER_MODELS) {
                recentOtherModelsBuffer.removeFirst()
            }
        }
    }

    private fun potentiallyEvaluate(epoch: Int, iteration: Int) {
        if (iteration < 20 || iteration % iterationsBeforeEvaluation == 0) {
            val evaluationScript = {
                val elapsedTime2 = System.currentTimeMillis() - start
                val extraElements2 = mapOf(
                    Pair("before or after averaging", "after"),
                    Pair("#peers included in current batch", newOtherModelBuffer.size.toString())
                )
                evaluationProcessor.evaluate(
                    iterTestFull,
                    network,
                    extraElements2,
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

    fun getNodeIndex(): Int {
        return nodeIndex
    }

    fun addNetworkMessage(from: Int, message: INDArray) {
        newOtherModelBufferTemp.last()[from] = message.dup()
    }

    fun printIterations() {
        network.setListeners(ScoreIterationListener(5))
    }

    fun getLabels(): List<String> {
        return labels
    }

    fun setCountPerPeer(countPerPeer: Map<Int, Int>) {
        this.countPerPeer = countPerPeer
    }
}
