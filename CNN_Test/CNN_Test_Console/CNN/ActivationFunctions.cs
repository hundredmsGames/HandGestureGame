using System;

namespace ConvNeuralNetwork
{
    static class ActivationFunctions
    {

        public static Tuple<Func<float, float>, Func<float, float>> GetActivationFuncs(ActivationType activationType)
        {
            switch (activationType)
            {
                case ActivationType.RELU:
                    return new Tuple<Func<float, float>, Func<float, float>>(ActivationFunctions.ReLu, ActivationFunctions.DerOfReLu);
                case ActivationType.SIGMOID:

                    return new Tuple<Func<float, float>, Func<float, float>>(ActivationFunctions.Sigmoid, ActivationFunctions.DerOfSigmoid);
                case ActivationType.SOFTMAX:

                    return new Tuple<Func<float, float>, Func<float, float>>(ActivationFunctions.Softmax, ActivationFunctions.DerOfSoftmax);
                case ActivationType.TANH:

                    return new Tuple<Func<float, float>, Func<float, float>>(ActivationFunctions.Tanh, ActivationFunctions.DerTanh);
                default:
                    //Throw exception for unrecognizeable method
                    return null;
            }
        }

        /// <summary>
        /// Rectified Linear Units: Max(x, 0)
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static float ReLu(float x)
        {
            return Math.Max(x, 0);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static float DerOfReLu(float x)
        {
            return (x > 0) ? 1f : 0f;
        }

        public static float Tanh(float x)
        {
            return 2f / (1f + (float)Math.Exp(-2f * x)) - 1f;
        }

        public static float DerTanh(float x)
        {
            float tanh = Tanh(x);

            return 1f - tanh * tanh;
        }

        public static float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        public static float DerOfSigmoid(float x)
        {
            return x * (1f - x);
        }

        public static float Softmax(float x)
        {

            return 0f;
        }

        public static float DerOfSoftmax(float x)
        {

            return 0f;
        }
    }
}
