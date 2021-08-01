package nl.tudelft.dfl

import kotlinx.serialization.Serializable

@Serializable
class ExperimentJSON {
    lateinit var name: String
    lateinit var fixedValues: Map<String, String>
    lateinit var variableValues: Map<String, List<String>>
}