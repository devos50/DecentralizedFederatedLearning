package nl.tudelft.dfl.configuration

import nl.tudelft.dfl.types.ModelPoisoningAttack

data class AttackConfiguration(
    var attack: ModelPoisoningAttack,
    var numAttackers: Int
) {
    companion object {
        fun defaultConfiguration() : AttackConfiguration {
            return AttackConfiguration(ModelPoisoningAttack.NONE, 0)
        }
    }

    fun copy(): AttackConfiguration {
        return AttackConfiguration(attack, numAttackers)
    }
}
