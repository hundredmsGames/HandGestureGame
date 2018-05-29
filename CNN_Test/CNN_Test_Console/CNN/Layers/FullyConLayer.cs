using System;
using MatrixLib;

namespace ConvNeuralNetwork
{
    class FullyConLayer : Layer
    {
        #region Variables

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

        public FullyConLayer(int[] layerTop, ActivationType activationHidden, ActivationType activationOutput) : base(LayerType.FULLY_CONNECTED)
        {
            hidLayers = new int[layerTop.Length - 2];
            weights = new Matrix[layerTop.Length - 1];
            biases = new Matrix[layerTop.Length - 1];
            layerOutputs = new Matrix[layerTop.Length - 1];

            this.inputNodes = layerTop[0];
            this.outputNodes = layerTop[layerTop.Length - 1];
            for (int i = 0; i < layerTop.Length - 2; i++)
            {
                this.hidLayers[i] = layerTop[i + 1];
            }

            for (int i = 0; i < layerOutputs.Length; i++)
            {
                layerOutputs[i] = new Matrix(layerTop[i], 1);
            }

            this.Output = new Matrix[1];
            this.Output[0] = new Matrix(outputNodes, 1);
            this.Output_d_E = new Matrix[1];
            this.Output_d_E[0] = new Matrix(outputNodes, 1);

            weights[0] = new Matrix(hidLayers[0], inputNodes);
            biases[0] = new Matrix(hidLayers[0], 1);
            weights[weights.Length - 1] = new Matrix(outputNodes, hidLayers[hidLayers.Length - 1]);
            biases[biases.Length - 1] = new Matrix(outputNodes, 1);


            Tuple<Func<float, float>, Func<float, float>> hiddenFuncs, outputFuncs;
            hiddenFuncs= ActivationFunctions.GetActivationFuncs(activationHidden);
            outputFuncs = ActivationFunctions.GetActivationFuncs(activationOutput);

            //set activation funcs
            this.activationHidden = hiddenFuncs.Item1;
            this.derOfActivationHidden = hiddenFuncs.Item2;
            this.activationOutput = outputFuncs.Item1;
            this.derOfActivationOutput = outputFuncs.Item2;
        }

        public override void Initialize()
        {
            base.Initialize();

            for (int i = 1; i <= weights.Length - 2; i++)
            {
                weights[i] = new Matrix(hidLayers[i], hidLayers[i - 1]);
                biases[i] = new Matrix(hidLayers[i], 1);
            }

            for (int i = 0; i < weights.Length; i++)
            {
                weights[i].Randomize();
            }

            for (int i = 0; i < biases.Length; i++)
            {
                biases[i].Randomize();
            }
        }

        #endregion

        #region Training Methods

        public override void FeedForward()
        {
            base.FeedForward();

            // Decrease dimension to 1
            fixedInput = DecreaseDimension(Input);

            for (int i = 0; i < layerOutputs.Length; i++)
            {
                if (i == 0)
                    layerOutputs[i] = weights[i] * fixedInput;
                else
                    layerOutputs[i] = weights[i] * layerOutputs[i - 1];

                layerOutputs[i] += biases[i];

                if (i == layerOutputs.Length - 1)
                    layerOutputs[i].Map(activationOutput);
                else
                    layerOutputs[i].Map(activationHidden);
            }

            Output[0] = layerOutputs[layerOutputs.Length - 1];

            if (OutputLayer != null)
            {
                OutputLayer.Input = new Matrix[1];
                OutputLayer.Input[0] = Output[0];
            }
        }

        public override void Backpropagation()
        {
            base.Backpropagation();

            Matrix net_d_E = null;
            Matrix w_d_net = null;
            Matrix w_d_E = null;
            Matrix out_d_net = null;
            Matrix out_d_E = null;

            for(int i = layerOutputs.Length - 1; i >= 0; i--)
            {
                // Multiply, derivative of lost function w.r.t output and
                // derivative of output (activation) w.r.t net
                if (i == layerOutputs.Length - 1)
                    net_d_E = Matrix.Multiply(layerOutputs[i] - Network.Target, Matrix.Map(layerOutputs[i], derOfActivationOutput));
                else
                    net_d_E = Matrix.Multiply(out_d_E, Matrix.Map(layerOutputs[i], derOfActivationHidden));


                //der of input to current layer w.r.t weight
                if (i != 0)
                    w_d_net = Matrix.Map(layerOutputs[i - 1], DerNetFunc);
                else
                    w_d_net = Matrix.Map(fixedInput, DerNetFunc);


                w_d_E = net_d_E * Matrix.Transpose(w_d_net);
              
                out_d_net = Matrix.Map(weights[i], DerNetFunc);

                weights[i] = weights[i] - (this.Network.LearningRate * w_d_E);

                out_d_E = Matrix.Transpose(out_d_net) * net_d_E;
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

        public static float DerNetFunc(float x)
        {
            return x;
        }

        #endregion
    }
}

