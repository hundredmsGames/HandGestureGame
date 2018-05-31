using System;
using MatrixLib;

namespace ConvNeuralNetwork
{
    class FC_Network : Layer
    {
        #region Variables

        FC_Layer[] layers;

        // -------------

        private int inputNodes;
        private int[] hidLayers;
        private int outputNodes;

        private Matrix fixedInput;
        private Matrix[] weights;
        private Matrix[] biases;

        private Matrix[] layerOutputs;

        private Func<float, float> activationHidden;
        private Func<float, float> derOfActivationHidden;

        private Func<float, float> activationOutput;
        private Func<float, float> derOfActivationOutput;

        public Matrix[] Weights { get => weights; set => weights = value; }
        public Matrix[] Biases { get => biases; set => biases = value; }

        #endregion

        #region Constructors

        public FC_Network(int[] topology, ActivationType[] activationTypes) : base(LayerType.FULLY_CONNECTED)
        {
            layers = new FC_Layer[topology.Length - 1];

            for(int i = 0; i < layers.Length; i++)
            {
                layers[i] = new FC_Layer(topology[i + 1], topology[i], activationTypes[i]);
                layers[i].Network = this.Network;
            }

            Output = new Matrix[1];
        }

        public override void Initialize()
        {
            base.Initialize();
        }

        #endregion

        #region Training Methods

        public override void FeedForward()
        {
            base.FeedForward();

            // Decrease dimension to 1
            layers[0].Input[0] = DecreaseDimension(Input);

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].FeedForward();
            }

            Output[0] = layers[layers.Length - 1].Output[0];
        }

        public override void Backpropagation()
        {
            base.Backpropagation();



            for (int i = layers.Length; i >= 0; i--)
            {

            }

            // Increase dimension back
            InputLayer.Output_d_E = IncreaseDimension(out_d_E);
        }

        private Matrix[] IncreaseDimension(Matrix oldMatrix)
        {
            Matrix[] increasedMatrix = new Matrix[Input.Length];
            int currIndex = 0;

            for (int ch = 0; ch < increasedMatrix.Length; ch++)
            {
                increasedMatrix[ch] = new Matrix(Input[0].rows, Input[0].cols);
                for (int r = 0; r < increasedMatrix[0].rows; r++)
                {
                    for (int c = 0; c < increasedMatrix[0].cols; c++)
                    {
                        increasedMatrix[ch][r, c] = oldMatrix[currIndex++, 0];
                    }
                }
            }

            return increasedMatrix;
        }

        private Matrix DecreaseDimension(Matrix[] oldMatrix)
        {
            int len = oldMatrix.Length * oldMatrix[0].rows * oldMatrix[0].cols;
            Matrix decreasedMatrix = new Matrix(len, 1);

            int currIndex = 0;
            for (int ch = 0; ch < oldMatrix.Length; ch++)
            {
                for (int r = 0; r < oldMatrix[0].rows; r++)
                {
                    for (int c = 0; c < oldMatrix[0].cols; c++)
                    {
                        decreasedMatrix[currIndex++, 0] = oldMatrix[ch][r, c];
                    }
                }
            }

            return decreasedMatrix;
        }

        #endregion
    }
}

