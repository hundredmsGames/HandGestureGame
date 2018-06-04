using System;
using MatrixLib;

namespace ConvNeuralNetwork
{
    static class ActivationFunctions
    {
        public static void GetActivationFuncs(
            ActivationType activationType,
            out Func<Matrix, Matrix> act,
            out Func<Matrix, Matrix> der
        )
        {
            switch (activationType)
            {
                case ActivationType.RELU:
                    act = ReLu;
                    der = DerOfReLu;
                    break;
                case ActivationType.SIGMOID:
                    act = Sigmoid;
                    der = DerOfSigmoid;
                    break;
                case ActivationType.SOFTMAX:
                    act = Softmax;
                    der = DerOfSoftmax;
                    break;
                case ActivationType.TANH:
                    act = Tanh;
                    der = DerTanh;
                    break;
                default:
                    //Throw exception for unrecognizeable method
                    act = der = null;
                    break;
            }
        }

        /// <summary>
        /// Rectified Linear Units: Max(x, 0)
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static Matrix ReLu(Matrix m)
        {
            return Matrix.Map(m, x => Math.Max(x, 0));
        }

        /// <summary>
        /// Derivative of ReLu
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static Matrix DerOfReLu(Matrix m)
        {
            return Matrix.Map(m, x => (x > 0) ? 1f : 0f);
        }

        public static Matrix Tanh(Matrix m)
        {
            return 2f / (1f + Matrix.Exp(-2f * m)) - 1f;
        }

        public static Matrix DerTanh(Matrix m)
        {
            Matrix tanh = Tanh(m);

            return 1f - Matrix.Multiply(tanh, tanh);
        }

        public static Matrix Sigmoid(Matrix m)
        {
            return 1f / (1f + Matrix.Exp(-m));
        }

        public static Matrix DerOfSigmoid(Matrix m)
        {
            return Matrix.Multiply(m, (1f - m));
        }

        public static float Sigmoid(float x)
        {
            return 1f / (1f + (float)Math.Exp(-x));
        }

        public static float DerOfSigmoid(float x)
        {
            return x * (1f - x);
        }

        public static Matrix Softmax(Matrix m)
        {

            return null;
        }

        public static Matrix DerOfSoftmax(Matrix x)
        {

            return null;
        }
    }
}
