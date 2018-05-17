using System;
using MatrixLib;

namespace FullyConnectedNN
{
    public class FCNN
    {
        #region Variables

        private int inputNodes;
        private int hiddenNodes;
        private int outputNodes;

        private Matrix weights_ih;
        private Matrix weights_ho;
        private Matrix bias_h;
        private Matrix bias_o;

        private Matrix input;
        private Matrix outs_out;
        private Matrix out_hid;

        private float learningRate;

        private Func<float, float> activationFunc;
        private Func<float, float> derOfActFunc;

        #endregion

        #region Constructors

        public FCNN(int inputNodes, int hiddenNodes, int outputNodes, float learningRate,
            Func<float, float> activationFunc, Func<float, float> derOfActivationFunc)
        {
            this.inputNodes = inputNodes;
            this.hiddenNodes = hiddenNodes;
            this.outputNodes = outputNodes;

            this.learningRate = learningRate;

            weights_ih = new Matrix(hiddenNodes, inputNodes);
            weights_ho = new Matrix(outputNodes, hiddenNodes);
            weights_ih.Randomize();
            weights_ho.Randomize();

            bias_h = new Matrix(this.hiddenNodes, 1);
            bias_o = new Matrix(this.outputNodes, 1);
            bias_h.Randomize();
            bias_o.Randomize();

            this.activationFunc = activationFunc;
            this.derOfActFunc = derOfActivationFunc;
        }

        #endregion

        #region Training Methods

        public Matrix FeedForward(Matrix input)
        {
            // Generating the hidden outputs.
            this.input = input;
            this.out_hid = this.weights_ih * this.input;
            this.out_hid += this.bias_h;
            this.out_hid.Map(activationFunc);

            // Generating the output's output.
            this.outs_out = this.weights_ho * this.out_hid;
            this.outs_out += this.bias_o;
            this.outs_out.Map(activationFunc);

            Console.WriteLine(outs_out.ToString());
            return this.outs_out;
        }

        private Matrix Backpropagation(Matrix target)
        {
            // Backpropagation Process
            Matrix neto_d_E = Matrix.Multiply(outs_out - target, Matrix.Map(outs_out, DerSigmoid));

            Matrix wo_d_neto = Matrix.Map(out_hid, DerNetFunc);

            Matrix wo_d_E = neto_d_E * Matrix.Transpose(wo_d_neto);

            Matrix outh_d_neto = Matrix.Map(weights_ho, DerNetFunc);

            weights_ho = weights_ho - (learningRate * wo_d_E);


            Matrix outh_d_E = Matrix.Transpose(outh_d_neto) * neto_d_E;

            Matrix neth_d_outh = Matrix.Map(out_hid, derOfActFunc);

            Matrix neth_d_E = Matrix.Multiply(outh_d_E, neth_d_outh);

            Matrix wh_d_neth = Matrix.Map(input, DerNetFunc);

            Matrix wh_d_E = wh_d_neth * Matrix.Transpose(neth_d_E);

            Matrix in_d_neth = Matrix.Map(weights_ih, DerNetFunc);

            Matrix in_d_E = Matrix.Transpose(in_d_neth) * neth_d_E;

            weights_ih = weights_ih - (learningRate * Matrix.Transpose(wh_d_E));
            Console.WriteLine(in_d_E.ToString());
            return in_d_E;
        }

        // If you want to get output, you need an extra parameter
        public Matrix Train(Matrix input, Matrix target)
        {
            FeedForward(input);

            return Backpropagation(target);
        }

        public float GetError(Matrix target, Matrix output)
        {
            // Calculate the error 
            // ERROR = (1 / 2) * (TARGETS - OUTPUTS)^2

            Matrix outputError = target - output;
            outputError = Matrix.Multiply(outputError, outputError) / 2f;

            float error = 0f;
            for (int i = 0; i < outputError.data.GetLength(0); i++)
                error += outputError.data[i, 0];

            return error;
        }

        #endregion

        #region Activation Funcs and Derivatives

        public static float Tanh(float x)
        {
            return 2f / (1f + (float) Math.Exp(-2f * x)) - 1f;
        }

        public static float DerTanh(float x)
        {
            float tanh = Tanh(x);

            return 1f - tanh * tanh;
        }

        public static float Sigmoid(float x)
        {
            return 1f / (float) (1f + Math.Exp(-x));
        }

        public static float DerSigmoid(float x)
        {
            return x * (1f - x);
        }

        public static float DerNetFunc(float x)
        {
            return x;
        }

        #endregion
    }
}

