using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MatrixLib;

namespace ConvNeuralNetwork
{
    class FC_Layer : Layer
    {
        #region Variables

        private Matrix weights;
        private Matrix biases;
        Func<float, float> activation;
        Func<float, float> derOfActivation;
        
        #endregion

        public FC_Layer(int inputNeurons, int nextLayerNeurons, ActivationType activationType) : base(LayerType.FC_LAYER)
        {
            this.weights = new Matrix(nextLayerNeurons, inputNeurons);
            this.biases  = new Matrix(nextLayerNeurons, 1);
            this.weights.Randomize();
            this.biases.Randomize();

            // TODO: This can be shorter with out parameters.
            Tuple<Func<float, float>, Func<float, float>> Funcs = ActivationFunctions.GetActivationFuncs(activationType);
            activation      = Funcs.Item1;
            derOfActivation = Funcs.Item2;

            Input           = new Matrix[1];
            Output          = new Matrix[1];
            Output_d_E      = new Matrix[1];
            Input[0]        = new Matrix(inputNeurons, 1);
            Output[0]       = new Matrix(nextLayerNeurons, 1);
            Output_d_E[0]   = new Matrix(nextLayerNeurons, 1);
        }

        public override void Initialize()
        {
            base.Initialize();
        }

        public override void FeedForward()
        {
            base.FeedForward();

            Output[0]  = weights * Input[0];
            Output[0] += biases;
            Output[0].Map(activation);

            // If this layer is last layer so there is no outputLayer.
            if(OutputLayer != null)
                OutputLayer.Input[0] = Output[0];
        }

        public override void Backpropagation()
        {
            base.Backpropagation();
            
            Matrix net_d_E      = Matrix.Multiply(Output_d_E[0], Matrix.Map(Output[0], derOfActivation));
            Matrix w_d_net      = Matrix.Map(Input[0], DerNetFunc);
            Matrix w_d_E        = net_d_E * Matrix.Transpose(w_d_net);
            Matrix out_d_net    = Matrix.Map(weights, DerNetFunc);
            Matrix out_d_E      = Matrix.Transpose(out_d_net) * net_d_E;
            weights             = weights - (this.Network.LearningRate * w_d_E);      

            // If this is first layer, don't need to backpropagate
            if(InputLayer != null)
                InputLayer.Output_d_E[0] = out_d_E;
        }

        public static float DerNetFunc(float x)
        {
            return x;
        }        
    }
}
