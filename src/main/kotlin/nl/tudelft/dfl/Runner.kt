package nl.tudelft.dfl

import mu.KotlinLogging
import nl.tudelft.dfl.*
import nl.tudelft.dfl.dataset.CustomDatasetIterator
import nl.tudelft.dfl.dataset.CustomDatasetType
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import java.io.File
import java.lang.Integer.max
import kotlin.random.Random

private val logger = KotlinLogging.logger("Runner")

private const val TEST_SET_SIZE = 10

class Runner {
    private lateinit var nodes: List<Node>
    private lateinit var nodeGetsFrom: Map<Int, List<Node>>
    private lateinit var nodeSendsTo: Map<Int, List<Node>>
    private var peersRR: MutableMap<Int, MutableList<Node>?> = HashMap()
    private var peersRing: MutableMap<Int, MutableList<Node>?> = HashMap()
    private var ringCounter: MutableMap<Int, Int> = HashMap()

    fun generateNetwork(
        architecture: (nnConfiguration: NNConfiguration, seed: Int, mode: NNConfigurationMode) -> MultiLayerConfiguration,
        nnConfiguration: NNConfiguration,
        seed: Int,
        mode: NNConfigurationMode,
    ): MultiLayerNetwork {
        val network = MultiLayerNetwork(architecture(nnConfiguration, seed, mode))
        network.init()
        return network
    }

    protected fun getDataSetIterators(
        inst: (iteratorConfiguration: DatasetIteratorConfiguration, seed: Long, dataSetType: CustomDatasetType, baseDirectory: File, behavior: Behaviors, transfer: Boolean) -> CustomDatasetIterator,
        datasetIteratorConfiguration: DatasetIteratorConfiguration,
        seed: Long,
        baseDirectory: File,
        behavior: Behaviors,
    ): List<CustomDatasetIterator> {
        val trainDataSetIterator = inst(
            DatasetIteratorConfiguration(
                datasetIteratorConfiguration.batchSize,
                datasetIteratorConfiguration.distribution,
                datasetIteratorConfiguration.maxTestSamples
            ),
            seed,
            CustomDatasetType.TRAIN,
            baseDirectory,
            behavior,
            false,
        )
        logger.debug { "Loaded trainDataSetIterator" }
        val testDataSetIterator = inst(
            DatasetIteratorConfiguration(
                BatchSizes.BATCH_200,
                datasetIteratorConfiguration.distribution.map { if (it == 0) 0 else TEST_SET_SIZE },
                datasetIteratorConfiguration.maxTestSamples
            ),
            seed + 1,
            CustomDatasetType.TEST,
            baseDirectory,
            behavior,
            false,
        )
        logger.debug { "Loaded testDataSetIterator" }
        val fullTestDataSetIterator = inst(
            DatasetIteratorConfiguration(
                BatchSizes.BATCH_200,
                List(datasetIteratorConfiguration.distribution.size) { datasetIteratorConfiguration.maxTestSamples.value },
                datasetIteratorConfiguration.maxTestSamples
            ),
            seed + 2,
            CustomDatasetType.FULL_TEST,
            baseDirectory,
            Behaviors.BENIGN,
            false,
        )
        logger.debug { "Loaded fullTestDataSetIterator" }
        return listOf(trainDataSetIterator, testDataSetIterator, fullTestDataSetIterator)
    }

    protected fun craftMessage(first: INDArray, behavior: Behaviors, random: Random): INDArray {
        return when (behavior) {
            Behaviors.BENIGN -> first
            Behaviors.NOISE -> craftNoiseMessage(first, random)
            Behaviors.LABEL_FLIP_2 -> first
            Behaviors.LABEL_FLIP_ALL -> first
        }
    }

    private fun craftNoiseMessage(first: INDArray, random: Random): INDArray {
        val numColumns = first.columns()
        return NDArray(Array(first.rows()) { FloatArray(numColumns) { random.nextFloat() / 2 - 0.2f } })
    }

    private fun shareModel(
        params: INDArray,
        trainConfiguration: TrainConfiguration,
        random: Random,
        nodeIndex: Int,
        countPerPeer: Map<Int, Int>
    ) {
        val message = craftMessage(params, trainConfiguration.behavior, random)
        when (trainConfiguration.communicationPattern) {
            CommunicationPatterns.ALL -> nodeSendsTo[nodeIndex]!!.forEach { it.addNetworkMessage(nodeIndex, message) }
            CommunicationPatterns.RANDOM -> nodes
                .filter { it.getNodeIndex() != nodeIndex }
                .random().addNetworkMessage(nodeIndex, message)
            CommunicationPatterns.RR -> {
                if (peersRR[nodeIndex].isNullOrEmpty()) {
                    peersRR[nodeIndex] = nodes.filter { it.getNodeIndex() != nodeIndex }.toMutableList()
                    val index = peersRR[nodeIndex]!!.indexOfFirst { it.getNodeIndex() > nodeIndex }
                    for (i in 0 until index) {
                        peersRR[nodeIndex]!!.add(peersRR[nodeIndex]!!.removeAt(0))
                    }
                }
                peersRR[nodeIndex]!!.removeAt(0).addNetworkMessage(nodeIndex, message)
            }
            CommunicationPatterns.RING -> {
                if (peersRing[nodeIndex].isNullOrEmpty() || peersRing[nodeIndex]!!.size < ringCounter[nodeIndex]!!) {
                    peersRing[nodeIndex] = nodes.filter { it.getNodeIndex() != nodeIndex }.toMutableList()
                    val index = peersRing[nodeIndex]!!.indexOfFirst { it.getNodeIndex() > nodeIndex }
                    for (i in 0 until index) {
                        peersRing[nodeIndex]!!.add(peersRing[nodeIndex]!!.removeAt(0))
                    }
                    ringCounter[nodeIndex] = 1
                }
                for (i in 0 until ringCounter[nodeIndex]!! - 1) {
                    peersRing[nodeIndex]!!.removeAt(0)
                }
                ringCounter[nodeIndex] = ringCounter[nodeIndex]!! * 2
                peersRing[nodeIndex]!!.removeAt(0).addNetworkMessage(nodeIndex, message)
            }
            CommunicationPatterns.RANDOM_3 -> {
                repeat(3) {
                    nodes
                        .filter { it.getNodeIndex() != nodeIndex }
                        .random().addNetworkMessage(nodeIndex, message)
                }
            }
        }
    }

    private fun getCountPerPeers(testConfig: List<MLConfiguration>, nodes: List<Node>): Map<Int, Map<Int, Int>> {
        return nodes.indices.map { i ->
            Pair(i, nodes.indices.map { j ->
                Pair(j, 3)
            }.toMap()
            )
        }.toMap()
    }

    fun performTest(
        baseDirectory: File,
        figureName: String,
        testConfig: List<MLConfiguration>,
        evaluationProcessor: EvaluationProcessor
    ) {
        val transfer = testConfig[0].trainConfiguration.transfer
        val fullFigureName = "$figureName - ${testConfig[0].trainConfiguration.gar.id} - ${if (transfer) "transfer" else "regular"}"
        logger.error { "Going to test: $fullFigureName" }

        // Initialize everything
        evaluationProcessor.newSimulation(figureName, testConfig, transfer)
        val start = System.currentTimeMillis()
        nodes = testConfig.mapIndexed { i, config ->
            Node(
                i,
                config,
                ::generateNetwork,
                ::getDataSetIterators,
                baseDirectory,
                evaluationProcessor,
                start,
                ::shareModel
            )
        }
        val numConnectedNodes = max(0, (testConfig[0].trainConfiguration.connectionRatio * nodes.size).toInt())
        logger.debug { "nodes: $nodes" }
        logger.debug { "nodes not benign: ${nodes.filter { it.behavior != Behaviors.BENIGN }}"}
        logger.debug { "numConnectedNodes: $numConnectedNodes" }
        nodeGetsFrom = nodes.map { node ->
            Pair(node.getNodeIndex(), listOf(nodes.filter { it.behavior != Behaviors.BENIGN }, nodes.filter { it.getNodeIndex() != node.getNodeIndex() && it.behavior == Behaviors.BENIGN }.shuffled().take(numConnectedNodes)).flatten())
        }.toMap()
        nodeSendsTo = nodes.map { node ->
            Pair(node.getNodeIndex(), nodeGetsFrom.filter { node in it.value }.keys.map { nodeIndex -> nodes.filter { it.getNodeIndex() == nodeIndex }.first() })
        }.toMap()
        logger.debug { "connected to: ${nodeSendsTo[0]}" }
        testConfig.forEachIndexed { i, _ ->
            ringCounter[i] = 1
        }
        val countPerPeers = getCountPerPeers(testConfig, nodes)
        nodes.forEachIndexed { i, node -> node.setCountPerPeer(countPerPeers.getValue(i)) }
        nodes[0].printIterations()

        // Perform <x> iterations
        var epochEnd = true
        var epoch = -1
        for (iteration in 0 until testConfig[0].trainConfiguration.maxIteration.value) {
            if (epochEnd) {
                epoch++
                logger.debug { "Epoch: $epoch" }
                epochEnd = false
            }
            logger.debug { "Iteration: $iteration" }

            nodes.forEach { it.applyNetworkBuffers() }
            val endEpochs = nodes.map { it.performIteration(epoch, iteration) }
            if (endEpochs.any { it }) epochEnd = true
        }
        logger.warn { "Test finished" }
    }

    fun simulate(
        baseDirectory: File,
        automationPart: Int,
    ) {
        val evaluationProcessor = EvaluationProcessor(
            baseDirectory,
            "simulated",
            listOf(
                "before or after averaging",
                "#peers included in current batch"
            )
        )
        try {
            val automation = loadAutomation(baseDirectory)
            logger.debug { "Automation loaded" }
            val (configs, figureNames) = generateConfigs(automation, automationPart)
            logger.debug { "Configs generated" }

            for (figure in configs.indices) {
                val figureName = figureNames[figure]
                val figureConfig = configs[figure]
                for (test in figureConfig.indices) {
                    val testConfig = figureConfig[test]
                    performTest(baseDirectory, figureName, testConfig, evaluationProcessor)
                }
            }
            evaluationProcessor.done()
            logger.error { "All tests finished" }
        } catch (e: Exception) {
            evaluationProcessor.error(e)
            e.printStackTrace()
        }
    }
}