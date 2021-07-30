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
    val baseDirectory = File(".")
    val runner = Runner()
    runner.simulate(baseDirectory, 0)
}