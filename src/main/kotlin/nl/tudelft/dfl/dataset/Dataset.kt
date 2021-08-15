package nl.tudelft.dfl.dataset

import nl.tudelft.dfl.configuration.DatasetIteratorConfiguration
import nl.tudelft.dfl.configuration.NNConfiguration
import nl.tudelft.dfl.configuration.NNConfigurationMode
import nl.tudelft.dfl.dataset.mnist.CustomMnistDataSetIterator
import nl.tudelft.dfl.types.*
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File

enum class Dataset(
    val id: String,
    val text: String,
    val defaultOptimizer: Optimizer,
    val defaultLearningRate: LearningRate,
    val defaultMomentum: Momentum,
    val defaultL2: L2Regularization,
    val defaultBatchSize: Int,
    val defaultIteratorDistribution: IteratorDistribution,
    val architecture: (nnConfiguration: NNConfiguration, seed: Int, mode: NNConfigurationMode) -> MultiLayerConfiguration,
    val inst: (iteratorConfiguration: DatasetIteratorConfiguration, seed: Long, dataSetType: CustomDatasetType, baseDirectory: File, behavior: Behavior) -> CustomDatasetIterator,
) {

    MNIST(
        "mnist",
        "MNIST",
        Optimizer.ADAM,
        LearningRate.RATE_1EM3,
        Momentum.NONE,
        L2Regularization.L2_5EM3,
        5,
        IteratorDistribution.DISTRIBUTION_MNIST_2,
        ::generateDefaultMNISTConfiguration,
        CustomMnistDataSetIterator::create,
    );
//    CIFAR10(
//        "cifar10",
//        "CIFAR-10",
//        Optimizers.ADAM,
//        LearningRates.RATE_1EM3,
//        Momentums.NONE,
//        L2Regularizations.L2_5EM3,
//        BatchSizes.BATCH_32,
//        IteratorDistributions.DISTRIBUTION_CIFAR_50,
//        ::generateDefaultCIFARConfiguration,
//        CustomCifar10DataSetIterator::create,
//    ),

//    TINYIMAGENET(
//        "tinyimagenet",
//        "Tiny ImageNet",
//        Optimizers.AMSGRAD,
//        LearningRates.SCHEDULE2,
//        Momentums.NONE,
//        L2Regularizations.L2_1EM4,
//        BatchSizes.BATCH_64,
//        IteratorDistributions.DISTRIBUTION_MNIST_1,
//        Runner::generateDefaultTinyImageNetConfiguration,
//        CustomMnistDataSetIterator::create
//    ),
//    HAR(
//        "har",
//        "HAR",
//        Optimizers.ADAM,
//        LearningRates.RATE_1EM3,
//        Momentums.NONE,
//        L2Regularizations.L2_1EM4,
//        BatchSizes.BATCH_32,
//        IteratorDistributions.DISTRIBUTION_HAR_100,
//        ::generateDefaultHARConfiguration,
//        HARDataSetIterator::create,
//    ),
//    WISDM(
//        "wisdm",
//        "WISDM",
////        Optimizers.NESTEROVS,
//        Optimizers.ADAM,
////        LearningRates.SCHEDULE1,
//        LearningRates.RATE_1EM3,
//        Momentums.NONE,
//        L2Regularizations.L2_1EM4,
//        BatchSizes.BATCH_5,
//        IteratorDistributions.DISTRIBUTION_WISDM_100,
//        ::generateDefaultWISDMConfiguration,
//        WISDMDataSetIterator::create,
//    ),

    companion object {
        fun load(id: String): Dataset {
            return values().firstOrNull { it.id == id } ?: throw Exception("Dataset ${id} not found")
        }
    }
}

fun generateDefaultMNISTConfiguration(
    nnConfiguration: NNConfiguration,
    seed: Int,
    mode: NNConfigurationMode,
): MultiLayerConfiguration {
    val numClasses = 10
    val layers = arrayOf<Layer>(
        ConvolutionLayer
            .Builder(intArrayOf(5, 5), intArrayOf(1, 1))
            .nOut(10)
            .build(),
        SubsamplingLayer
            .Builder(SubsamplingLayer.PoolingType.MAX, intArrayOf(2, 2), intArrayOf(2, 2))
            .build(),
        ConvolutionLayer
            .Builder(intArrayOf(5, 5), intArrayOf(1, 1))
            .nOut(50)
            .build(),
        SubsamplingLayer
            .Builder(SubsamplingLayer.PoolingType.MAX, intArrayOf(2, 2), intArrayOf(2, 2))
            .build(),
        OutputLayer
            .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(numClasses)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .hasBias(false)
            .build()
    )
    return NeuralNetConfiguration.Builder()
        .seed(seed.toLong())
        .activation(Activation.LEAKYRELU)
        .weightInit(WeightInit.RELU)
        .l2(nnConfiguration.l2.value)
        .updater(nnConfiguration.optimizer.inst(nnConfiguration.learningRate))
        .list()
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[0]).build()
            } else {
                layers[0]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[1]).build()
            } else {
                layers[1]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[2]).build()
            } else {
                layers[2]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[3]).build()
            } else {
                layers[3]
            }
        )
        .layer(layers[4])
        .setInputType(InputType.convolutionalFlat(28, 28, 1))
        .build()
}