package nl.tudelft.dfl

import mu.KotlinLogging
import nl.tudelft.dfl.configuration.RunConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.IEvaluation
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.*

private const val DATE_PATTERN = "yyyy-MM-dd_HH.mm.ss"
private val DATE_FORMAT = SimpleDateFormat(DATE_PATTERN, Locale.US)
private val logger = KotlinLogging.logger("EvaluationProcessor")

class EvaluationProcessor(
    baseDirectory: File,
) {
    private val configurationHeader = arrayOf(
        "name",
        "simulationIndex",
        "transfer",
        "dataset",
        "optimizer",
        "learning rate",
        "momentum",
        "l2",

        "batchSize",
        "iteratorDistribution",
        "maxTestSamples",

        "gar",
        "communicationPattern",
        "behavior",
        "numEpochs",
        "slowdown",
        "joiningLate",
        "iterationsBeforeEvaluation",
        "iterationsBeforeSending",

        "local model poisoning attack",
        "#attackers"
    ).joinToString(",")

    @Transient
    private val configurationLines = arrayListOf(configurationHeader)

    private val evaluationHeader = arrayOf(
        "name",
        "gar",
        "simulationIndex",
        "elapsedTime",
        "epoch",
        "iteration",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "gmeasure",
        "mcc",
        "score",
        "before or after averaging",
        "#peers included in current batch"
    )

    @Transient
    private val evaluationLines = arrayListOf(evaluationHeader)
    private val fileDirectory = File(baseDirectory.path, "evaluations")
    private val fileResults = File(fileDirectory, "evaluation-${DATE_FORMAT.format(Date())}.csv")
    private var fileMeta = File(fileDirectory, "evaluation-${DATE_FORMAT.format(Date())}.meta.csv")
    private lateinit var currentName: String
    private lateinit var garName: String
    private val startTime: Long

    init {
        if (!fileDirectory.exists()) {
            fileDirectory.mkdirs()
        }
        fileResults.createNewFile()
        fileMeta.createNewFile()

        if (fileLog == null) {
            fileLog = File(fileDirectory, "evaluation-${DATE_FORMAT.format(Date())}.txt")
            fileLog!!.createNewFile()
        }

        startTime = System.currentTimeMillis()
    }

    private fun convertToCSV(data: Array<String>): String {
        return data.joinToString(", ") { escapeSpecialCharacters(it) }
    }

    private fun escapeSpecialCharacters(data: String): String {
        return if (data.contains(",") || data.contains("\"") || data.contains("'")) {
            "\"" + data.replace("\"", "\"\"") + "\""
        } else {
            data.replace("\\R".toRegex(), " ")
        }
    }

    fun writeConfigurations(
        name: String,
        runConfiguration: RunConfiguration,
        transfer: Boolean
    ) {
        this.currentName = name
        this.garName = runConfiguration.trainConfiguration.gar.id
        listOf(runConfiguration).forEachIndexed { index, configuration ->
            val nnConfiguration = configuration.nnConfiguration
            val datasetIteratorConfiguration = configuration.datasetIteratorConfiguration
            val trainConfiguration = configuration.trainConfiguration
            val modelPoisoningConfiguration = configuration.attackConfiguration
            val line = arrayOf(
                name,
                index.toString(),
                transfer,
                configuration.dataset.id,

                nnConfiguration.optimizer.id,
                nnConfiguration.learningRate.id,
                nnConfiguration.momentum.id,
                nnConfiguration.l2.text,

                datasetIteratorConfiguration.batchSize,
                datasetIteratorConfiguration.distribution.toString().replace(", ", "-"),
                datasetIteratorConfiguration.maxTestSamples,

                trainConfiguration.gar.id,
                trainConfiguration.communicationPattern.id,
                trainConfiguration.behavior.id,
                trainConfiguration.maxIterations,
                trainConfiguration.iterationsBeforeEvaluation,
                trainConfiguration.iterationsBeforeSending,

                modelPoisoningConfiguration.attack.id,
                modelPoisoningConfiguration.numAttackers,
            ).joinToString(",")

            configurationLines.add(line)
        }
        PrintWriter(fileMeta).use { pw -> configurationLines.forEach(pw::println) }
    }

    private fun call(
        evaluations: Array<out IEvaluation<*>>,
        simulationIndex: Int,
        score: Double,
        elapsedTime: Long,
        iterations: Int,
        epoch: Int
    ): String {
        val accuracy = evaluations[0].getValue(Evaluation.Metric.ACCURACY)
        val f1 = evaluations[0].getValue(Evaluation.Metric.F1)
        val precision = evaluations[0].getValue(Evaluation.Metric.PRECISION)
        val recall = evaluations[0].getValue(Evaluation.Metric.RECALL)
        val gMeasure = evaluations[0].getValue(Evaluation.Metric.GMEASURE)
        val mcc = evaluations[0].getValue(Evaluation.Metric.MCC)
        val dataLineElements = arrayOf(
            this.currentName,
            this.garName,
            simulationIndex.toString(),
            elapsedTime.toString(),
            epoch.toString(),
            iterations.toString(),
            accuracy.toString(),
            f1.toString(),
            precision.toString(),
            recall.toString(),
            gMeasure.toString(),
            mcc.toString(),
            score.toString()
        )
        evaluationLines.add(Array(dataLineElements.size) { "" })
        System.arraycopy(dataLineElements, 0, evaluationLines.last(), 0, dataLineElements.size)
        PrintWriter(fileResults).use { pw ->
            evaluationLines
                .map(::convertToCSV)
                .forEach(pw::println)
        }
        return convertToCSV(evaluationLines.last())
    }

    fun error(e: Exception) {
        synchronized(evaluationLines) {
            evaluationLines.add(arrayOf(e.stackTraceToString()))

            PrintWriter(fileResults).use { pw ->
                evaluationLines
                    .map(::convertToCSV)
                    .forEach(pw::println)
            }
        }
    }

    fun evaluate(
        testDataSetIterator: DataSetIterator,
        network: MultiLayerNetwork,
        elapsedTime: Long,
        iterations: Int,
        epoch: Int,
        simulationIndex: Int,
        logging: Boolean
    ): String {
        testDataSetIterator.reset()
        val evaluations = arrayOf(Evaluation())

        logger.d(logging) {
            "Starting evaluation, #iterations = $iterations"
        }
        network.doEvaluation(testDataSetIterator, *evaluations)

        for (evaluation in evaluations) {
            logger.d(logging) { "${evaluation.javaClass.simpleName}:\n${evaluation.stats(false, true)}" }
        }
        return call(evaluations, simulationIndex, network.score(), elapsedTime, iterations, epoch)
    }

    companion object {
        var fileLog : File? = null
        private val logLines = arrayListOf<String>()

        fun log(message: String) {
            synchronized(logLines) {
                logLines.add(message)
                if (fileLog != null) {
                    PrintWriter(fileLog!!).use { pw ->
                        logLines
                            .forEach(pw::println)
                    }
                }
            }
        }
    }
}
