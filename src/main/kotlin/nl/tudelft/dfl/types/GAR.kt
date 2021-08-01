package nl.tudelft.dfl.types

import nl.tudelft.dfl.gar.*

enum class GAR(
    val id: String,
    val text: String,
    val obj: AggregationRule,
) {
    NONE("none", "None", NoAveraging()),
    AVERAGE("average", "Simple average", Average()),
    MEDIAN("median", "Median", Median()),
    KRUM("krum", "Krum (b=1)", Krum(4)),
    BRIDGE("bridge", "Bridge (b=1)", Bridge(4)),
    MOZI("mozi", "Mozi (frac=0.5)", Mozi(0.5));

    companion object {
        fun load(id: String): GAR {
            return values().firstOrNull { it.id == id } ?: throw Exception("GAR ${id} not found")
        }
    }
}