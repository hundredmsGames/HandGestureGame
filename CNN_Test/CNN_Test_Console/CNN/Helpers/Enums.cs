
namespace ConvNeuralNetwork
{
    enum LayerType:byte
    {
        INPUT,
        CONVOLUTIONAL,
        MAXPOOLING,
        FULLY_CONNECTED,
        FC_INPUT,
        FC_HIDDEN,
        FC_OUTPUT
    }

    enum ActivationType : byte
    {
        RELU,
        SIGMOID,
        SOFTMAX,
        TANH
    }
}
