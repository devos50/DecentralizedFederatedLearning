package nl.tudelft.dfl.types

import nl.tudelft.dfl.dataset.Dataset

enum class IteratorDistribution(val id: String, val text: String, val value: IntArray) {
    DISTRIBUTION_MNIST_1("mnist_100", "MNIST 100", intArrayOf(100, 100, 100, 100, 100, 100, 100, 100, 100, 100)),
    DISTRIBUTION_MNIST_2("mnist_500", "MNIST 500", intArrayOf(500, 500, 500, 500, 500, 500, 500, 500, 500, 500)),
    DISTRIBUTION_MNIST_3(
        "mnist_0_to_7_with_100",
        "MNIST 0 to 7 with 100",
        intArrayOf(100, 100, 100, 100, 100, 100, 100, 0, 0, 0)
    ),
    DISTRIBUTION_MNIST_4(
        "mnist_4_to_10_with_100",
        "MNIST 4 to 10 with 100",
        intArrayOf(0, 0, 0, 0, 100, 100, 100, 100, 100, 100)
    ),
    DISTRIBUTION_MNIST_5("mnist_7_to_4_with_100", "MNIST 0 to 7 with 100", intArrayOf(100, 100, 100, 0, 0, 0, 0, 100, 100, 100)),
    DISTRIBUTION_CIFAR_50("cifar_50", "CIFAR 50", intArrayOf(50, 50, 50, 50, 50, 50, 50, 50, 50, 50)),
    DISTRIBUTION_HAR_100("har_100", "HAR 100", intArrayOf(100, 100, 100, 100, 100, 100)),
    DISTRIBUTION_WISDM_100("wisdm_100", "WISDM 100", intArrayOf(100, 100, 100, 100, 100, 100));

    companion object {
        fun load(id: String): IteratorDistribution {
            return values().firstOrNull { it.id == id } ?: throw Exception("IteratorDistribution ${id} not found")
        }
    }
}