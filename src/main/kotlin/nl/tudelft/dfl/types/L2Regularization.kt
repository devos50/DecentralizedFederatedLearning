package nl.tudelft.dfl.types

enum class L2Regularization(val id: String, val text: String, val value: Double) {
    L2_5EM3("l2_5em3", "5e-3", 5e-3),
    L2_1EM4("l2_1em4", "1e-4", 1e-4);

    companion object {
        fun load(id: String): L2Regularization {
            return values().firstOrNull { it.id == id } ?: throw Exception("L2Regularization ${id} not found")
        }
    }
}