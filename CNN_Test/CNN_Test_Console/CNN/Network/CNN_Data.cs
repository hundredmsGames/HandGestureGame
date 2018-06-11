using MatrixLib;
using System.Collections.Generic;

namespace ConvNeuralNetwork
{
    class CNN_Data
    {
        #region Variables
        private List<Description> descriptions;

        List<Matrix[,]> kernels;

        List<Matrix> ker_biases;

        List<Matrix> weights;

        List<Matrix> biases;
        #endregion

        #region Properties
       
        public List<Description> Descriptions { get => descriptions; set => descriptions = value; }
        public List<Matrix[,]> Kernels { get => kernels; set => kernels = value; }
        public List<Matrix> Weights { get => weights; set => weights = value; }
        public List<Matrix> Biases { get => biases; set => biases = value; }
        public List<Matrix> Ker_Biases { get => ker_biases; set => ker_biases = value; }

        #endregion

        #region CTOR
        public CNN_Data()
        {
            descriptions = new List<Description>();
            kernels = new List<Matrix[,]>();
            weights = new List<Matrix>();
            biases = new List<Matrix>();
            ker_biases = new List<Matrix>();
        }
        #endregion
    }
}
