package nl.tudelft.dfl

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import mu.KotlinLogging
import nl.tudelft.dfl.configuration.RunConfiguration
import nl.tudelft.dfl.dataset.Dataset
import nl.tudelft.dfl.types.*
import java.io.File

private val logger = KotlinLogging.logger("Experiment")

class Experiment(experimentFilePath: String) {
    var json: ExperimentJSON
    private lateinit var nodes: List<Node>
    val configurations: ArrayList<RunConfiguration> = ArrayList()

    init {
        val file = File(experimentFilePath)
        val string = file.readLines().joinToString("")
        json = Json.decodeFromString(string)
    }

    private fun generateRunConfigurations() {
        for((variableSettingName, values) in json.variableValues) {
            for(variableValue in values) {
                val configuration = RunConfiguration.defaultConfiguration(Dataset.MNIST)
                if(variableSettingName == "gar") {
                    configuration.trainConfiguration.gar = GAR.load(variableValue)
                }

                if(json.fixedValues.contains("dataset")) {
                    configuration.dataset = Dataset.load(json.fixedValues.get("dataset")!!)
                }
                if(json.fixedValues.contains("batchSize")) {
                    configuration.datasetIteratorConfiguration.batchSize = json.fixedValues.get("batchSize")!!.toInt()
                }
                if(json.fixedValues.contains("iteratorDistribution")) {
                    var distribution: IntArray? = null
                    val d = json.fixedValues.getValue("iteratorDistribution")
                    if (d.startsWith('[')) {
                        distribution = d.substring(1, d.length - 1).split(", ").map { it.toInt() }.toIntArray()
                    } else {
                        distribution = IteratorDistribution.load(d).value
                    }
                    configuration.datasetIteratorConfiguration.distribution = distribution.toList()
                }
                if(json.fixedValues.contains("maxTestSamples")) {
                    configuration.datasetIteratorConfiguration.maxTestSamples = json.fixedValues.get("maxTestSamples")!!.toInt()
                }
                if(json.fixedValues.contains("optimizer")) {
                    configuration.nnConfiguration.optimizer = Optimizer.load(json.fixedValues.get("optimizer")!!)
                }
                if(json.fixedValues.contains("learningRate")) {
                    configuration.nnConfiguration.learningRate = LearningRate.load(json.fixedValues.get("learningRate")!!)
                }
                if(json.fixedValues.contains("momentum")) {
                    configuration.nnConfiguration.momentum = Momentum.load(json.fixedValues.get("momentum")!!)
                }
                if(json.fixedValues.contains("l2Regularization")) {
                    configuration.nnConfiguration.l2 = L2Regularization.load(json.fixedValues.get("l2Regularization")!!)
                }
                if(json.fixedValues.contains("maxIterations")) {
                    configuration.trainConfiguration.maxIterations = json.fixedValues.get("maxIterations")!!.toInt()
                }
                if(json.fixedValues.contains("gar") && variableSettingName != "gar") {
                    configuration.trainConfiguration.gar = GAR.load(json.fixedValues.get("gar")!!)
                }
                if(json.fixedValues.contains("communicationPattern")) {
                    configuration.trainConfiguration.communicationPattern = CommunicationPattern.load(json.fixedValues.get("communicationPattern")!!)
                }
                if(json.fixedValues.contains("behavior")) {
                    configuration.trainConfiguration.behavior = Behavior.load(json.fixedValues.get("behavior")!!)
                }
                if(json.fixedValues.contains("modelPoisoningAttack")) {
                    configuration.attackConfiguration.attack = ModelPoisoningAttack.load(json.fixedValues.get("modelPoisoningAttack")!!)
                }
                if(json.fixedValues.contains("numAttackers")) {
                    configuration.attackConfiguration.numAttackers = json.fixedValues.get("numAttackers")!!.toInt()
                }
                if(json.fixedValues.contains("numNodes")) {
                    configuration.numNodes = json.fixedValues.get("numNodes")!!.toInt()
                }

                configurations.add(configuration)
            }
        }
    }

    fun run() {
        generateRunConfigurations()
        val baseDirectory = File(".")
        val evaluationProcessor = EvaluationProcessor(baseDirectory)

        for(configuration in configurations) {
            val transfer = configuration.trainConfiguration.transfer
            val fullFigureName = "$json.name - ${configuration.trainConfiguration.gar.id} - ${if (transfer) "transfer" else "regular"}"
            logger.error { "Going to test: $fullFigureName" }

            val start = System.currentTimeMillis()

            nodes = List(configuration.numNodes) {
                Node(
                    it,
                    configuration,
                    baseDirectory,
                    evaluationProcessor,
                    start,
                )
            }

            evaluationProcessor.writeConfigurations(json.name, nodes, transfer)

            // connect peers to each other
            nodes.forEach() {
                it.neighbours = nodes.filter { it2 -> it != it2 }
            }

            // Perform <x> iterations
            var epochEnd = true
            var epoch = -1
            for (iteration in 0 until configuration.trainConfiguration.maxIterations) {
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
    }
}