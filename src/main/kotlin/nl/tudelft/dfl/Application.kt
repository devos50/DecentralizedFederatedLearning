package nl.tudelft.dfl

fun main() {
    val experiment = Experiment("experiment.json")
    experiment.run()
    print("Running experiment: ${experiment}")
}