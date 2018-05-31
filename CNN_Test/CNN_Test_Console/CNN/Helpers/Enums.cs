
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
        NONE,
        RELU,
        SIGMOID,
        SOFTMAX,
        TANH
    }
}
