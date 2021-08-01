package nl.tudelft.dfl.types

enum class Behavior(val id: String, val text: String) {
    BENIGN("benign", "Benign"),
    NOISE("noise", "Noise"),
    LABEL_FLIP_2("label_flip_2", "Label flip 2"),
    LABEL_FLIP_ALL("label_flip_all", "Label flip all");

    companion object {
        fun load(id: String): Behavior {
            return values().firstOrNull { it.id == id } ?: throw Exception("Behavior ${id} not found")
        }
    }
}