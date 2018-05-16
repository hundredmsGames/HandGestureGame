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
        private static double ReLu(double x)
        {
            return Math.Max(x, 0);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private static double DerOfReLu(double x)
        {
            return (x > 0) ? 1.0 : 0.0;
        }
    }
}
