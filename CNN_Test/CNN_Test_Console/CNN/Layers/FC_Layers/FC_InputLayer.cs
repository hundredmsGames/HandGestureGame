using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MatrixLib;

namespace ConvNeuralNetwork
{
    class FC_InputLayer : Layer
    {
        #region
        private Matrix weights;
        private Matrix biases;
        Func<float, float> activation;
        Func<float, float> derOfActivation;
        #endregion

        public FC_InputLayer(int inputNeurons, int nextLayerNeuronCount, ActivationType activationType) : base(LayerType.FC_INPUT)
        {
            this.weights = new Matrix(nextLayerNeuronCount, inputNeurons);
            this.weights.Randomize();
            this.biases = new Matrix(nextLayerNeuronCount, 1);
            this.biases.Randomize();

            Tuple<Func<float, float>, Func<float, float>> Funcs = ActivationFunctions.GetActivationFuncs(activationType);
            activation = Funcs.Item1;
            derOfActivation = Funcs.Item2;

            Input = new Matrix[1];
            Input[0] = new Matrix(inputNeurons, 1);
            Output = new Matrix[1];
            Output[0] = new Matrix(nextLayerNeuronCount, 1);
        }

        public override void Backpropagation()
        {
            base.Backpropagation();
            Matrix net_d_E = null;
            Matrix w_d_net = null;
            Matrix w_d_E = null;
            Matrix out_d_net = null;
            Matrix out_d_E = null;


            net_d_E = Matrix.Multiply(out_d_E, Matrix.Map(layerOutputs[i], derOfActivationHidden));


            //der of input to current layer w.r.t weight

            w_d_net = Matrix.Map(Input[0], DerNetFunc);


            w_d_E = net_d_E * Matrix.Transpose(w_d_net);

            out_d_net = Matrix.Map(weights, DerNetFunc);

            weights = weights - (this.Network.LearningRate * w_d_E);

            out_d_E = Matrix.Transpose(out_d_net) * net_d_E;

            // Increase dimension back
            this.InputLayer.Output_d_E[0] = out_d_E;

        }

        public override void FeedForward()
        {
            base.FeedForward();

            Output[0] = weights * Input[0];
            Output[0] += biases;
            Output[0].Map(activation);

            OutputLayer.Input = new Matrix[1];
            OutputLayer.Input[0] = Output[0];
        }
        public static float DerNetFunc(float x)
        {
            return x;
        }
        public override void Initialize()
        {
            base.Initialize();
        }
    }
}
