using MatrixLib;

namespace ConvNeuralNetwork
{
    class MaxPoolingLayer : Layer
    {
        #region Variables

        private int kernel_size;
        private int stride;

        #endregion

        #region Constructors

        public MaxPoolingLayer(int kernel_size, int stride) : base(LayerType.MAXPOOLING)
        {
            this.Kernel_Size = kernel_size;
            this.Stride = stride;
        }

        #endregion

        #region Properties

        public int Kernel_Size
        {
            get { return kernel_size; }
            protected set { kernel_size = value; }
        }

        public int Stride
        {
            get { return stride; }
            protected set { stride = value; }
        }

        #endregion
    }
}
