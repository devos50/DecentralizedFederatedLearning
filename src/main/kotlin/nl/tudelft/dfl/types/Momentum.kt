package nl.tudelft.dfl.types

enum class Momentum(val id: String, val text: String, val value: Double?) {
    NONE("none", "none", null),
    MOMENTUM_1EM3("momentum_1em3", "1e-3", 1e-3);

    companion object {
        fun load(id: String): Momentum {
            return values().firstOrNull { it.id == id } ?: throw Exception("Momentum ${id} not found")
        }
    }
}