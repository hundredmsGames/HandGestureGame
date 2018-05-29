
namespace ConvNeuralNetwork
{
    enum LayerType:byte
    {
        INPUT,
        CONVOLUTIONAL,
        MAXPOOLING,
        FULLY_CONNECTED
    }

    enum ActivationType : byte
    {
        RELU,
        SIGMOID,
        SOFTMAX,
        TANH
    }
}
