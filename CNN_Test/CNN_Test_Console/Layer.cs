using MatrixLib;

namespace ConvNeuralNetwork
{
    class Layer
    {
        private Matrix input;
        private Matrix output;

        #region Get-Set

        public Matrix Input
        {
            get { return input; }
            protected set { input = value; }
        }

        public Matrix Output
        {
            get { return output; }
            protected set { output = value; }
        }

        #endregion
    }
}
