package nl.tudelft.dfl.types

enum class CommunicationPattern(val id: String, val text: String) {
    ALL("all", "Send the model update to all peers"),
    RANDOM("random", "Send the model update to a random peer");

    companion object {
        fun load(id: String): CommunicationPattern {
            return values().firstOrNull { it.id == id } ?: throw Exception("Communication pattern ${id} not found")
        }
    }
}