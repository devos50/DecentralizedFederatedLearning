package nl.tudelft.dfl.types

import nl.tudelft.dfl.attack.PoisoningAttack
import nl.tudelft.dfl.attack.NoAttack

enum class ModelPoisoningAttack(val id: String, val text: String, val obj: PoisoningAttack) {
    NONE("none", "none", NoAttack());
//    FANG_2020_TRIMMED_MEAN("fang_2020_trimmed_mean", "Fang 2020 (trimmed mean)", Fang2020TrimmedMean(2)),
//    FANG_2020_KRUM("fang_2020_krum", "Fang 2020 (krum)", Fang2020Krum(2)),
//    NOISE("noise", "Noise", Noise())

    companion object {
        fun load(id: String): ModelPoisoningAttack {
            return values().firstOrNull { it.id == id } ?: throw Exception("Poisoning attack ${id} not found")
        }
    }
}