using System;
using MatrixLib;

namespace ConvNeuralNetwork
{
    class FC_Network : Layer
    {
        #region Variables

        FC_Layer[] layers;

        #endregion

        #region Constructors

        public FC_Network(int[] topology, ActivationType[] activationTypes) : base(LayerType.FULLY_CONNECTED)
        {
            layers = new FC_Layer[topology.Length - 1];

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new FC_Layer(topology[i], topology[i + 1], activationTypes[i]);

                if (i > 0)
                {
                    layers[i].InputLayer = layers[i - 1];
                }
            }
            for (int i = 0; i < layers.Length - 1; i++)
            {
                layers[i].OutputLayer = layers[i + 1];
            }
            layers[0].InputLayer = this;

            Output = new Matrix[1];
            Output_d_E = new Matrix[1];
        }

        public override void Initialize()
        {
            base.Initialize();

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].Network = this.Network;
            }
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

            //layers[layers.Length - 1].Output_d_E[0] = layers[layers.Length - 1].Output[0] - Network.Target;

            layers[layers.Length - 1].Output_d_E[0] = DerivativeOfCrossEntropy(
                layers[layers.Length - 1].Output[0],
                Network.Target
            );

            for (int i = layers.Length - 1; i >= 0; i--)
            {
                layers[i].Backpropagation();
            }

            // Increase dimension back
            InputLayer.Output_d_E = IncreaseDimension(Output_d_E[0]);
        }

        

        public Matrix DerivativeOfCrossEntropy(Matrix p, Matrix c)
        {
            // constant for divide by zero error. 
            return -(c / (p + 0.00000000000001)) + (1 - c) / (1 - p + 0.00000000000001);
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

        #region Properties

        public FC_Layer[] Layers
        {
            get { return layers; }
            set { layers = value; }
        }

        #endregion
    }
}

