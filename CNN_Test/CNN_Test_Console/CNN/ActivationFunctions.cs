using System;

namespace ConvNeuralNetwork
{
    static class ActivationFunctions
    {
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

        public static float Sigmoid(float x)
        {

            return 0f;
        }

        public static float DerOfSigmoid(float x)
        {

            return 0f;
        }

        public static float Tanh(float x)
        {

            return 0f;
        }

        public static float DerOfTanh(float x)
        {

            return 0f;
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
