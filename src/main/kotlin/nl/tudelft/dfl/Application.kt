package nl.tudelft.dfl

import nl.tudelft.dfl.demo.nl.tudelft.dfl.Runner
import java.io.File

fun main() {
    val baseDirectory = File(".")
    val runner = Runner()
    runner.simulate(baseDirectory, 0)
}