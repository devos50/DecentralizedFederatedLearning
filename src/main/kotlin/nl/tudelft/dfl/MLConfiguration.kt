package nl.tudelft.dfl

data class MLConfiguration(
    val dataset: Datasets,
    val datasetIteratorConfiguration: DatasetIteratorConfiguration,
    val nnConfiguration: NNConfiguration,
    val trainConfiguration: TrainConfiguration,
    val modelPoisoningConfiguration: ModelPoisoningConfiguration
)

data class DatasetIteratorConfiguration(
    val batchSize: BatchSizes,
    val distribution: List<Int>,
    val maxTestSamples: MaxTestSamples
)

data class NNConfiguration(
    val optimizer: Optimizers,
    val learningRate: LearningRates,
    val momentum: Momentums?,
    val l2: L2Regularizations
)

data class TrainConfiguration(
    val maxIteration: MaxIterations,
    val gar: GARs,
    val communicationPattern: CommunicationPatterns,
    val behavior: Behaviors,
    val slowdown: Slowdowns,
    val joiningLate: TransmissionRounds,
    val iterationsBeforeEvaluation: Int,
    val iterationsBeforeSending: Int,
    val transfer: Boolean,
    val connectionRatio: Double,
    val latency: Int
)

data class ModelPoisoningConfiguration(
    val attack: ModelPoisoningAttacks,
    val numAttackers: NumAttackers
)
