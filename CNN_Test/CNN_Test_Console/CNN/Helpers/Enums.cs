
namespace ConvNeuralNetwork
{
    enum LayerType:byte
    {
        INPUT,
        CONVOLUTIONAL,
        MAXPOOLING,
        FULLY_CONNECTED,
        FC_LAYER
    }

    enum ActivationType : byte
    {
        RELU,
        SIGMOID,
        SOFTMAX,
        TANH
    }
}
