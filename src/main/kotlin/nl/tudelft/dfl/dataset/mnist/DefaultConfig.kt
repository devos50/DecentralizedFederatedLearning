package nl.tudelft.dfl.dataset.mnist

import nl.tudelft.dfl.NNConfiguration
import nl.tudelft.dfl.NNConfigurationMode
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

fun generateDefaultMNISTConfiguration(
    nnConfiguration: NNConfiguration,
    seed: Int,
    mode: NNConfigurationMode,
): MultiLayerConfiguration {
    val numClasses = if (mode == NNConfigurationMode.TRANSFER) 26 else 10
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