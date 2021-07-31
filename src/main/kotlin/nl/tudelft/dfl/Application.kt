package nl.tudelft.dfl

import java.io.File

fun main() {
    val baseDirectory = File(".")
    val runner = Runner()
    runner.simulate(baseDirectory, 0)
}