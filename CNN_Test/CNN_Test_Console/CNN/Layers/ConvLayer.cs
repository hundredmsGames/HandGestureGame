using MatrixLib;

namespace ConvNeuralNetwork
{
    class ConvLayer : Layer
    {
        #region Variables

        private int filters;
        private int kernel_size;
        private int stride;
        private int padding;
        private Matrix[] kernels;

        #endregion

        #region Constructors

        public ConvLayer(int filters, int kernel_size, int stride, int padding = 0) : base(LayerType.CONVOLUTIONAL)
        {
            this.Filters = filters;
            this.Kernel_Size = kernel_size;
            this.Stride = stride;
            this.Padding = padding;

            // Initialize kernel
            kernels = new Matrix[filters];
            for (int i = 0; i < filters; i++)
            {
                kernels[i] = new Matrix(kernel_size, kernel_size);
                kernels[i].Randomize();
            }
        }

        #endregion

        #region Methods

        public override void FeedForward()
        {
            base.FeedForward();
        }

        public override void Backpropagation()
        {
            base.Backpropagation();
        }

        #endregion

        #region Properties

        public int Filters
        {
            get { return filters; }
            protected set { filters = value; }
        }

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

        public Matrix[] Kernels
        {
            get { return kernels; }
            protected set { kernels = value; }
        }

        public int Padding { get => padding; set => padding = value; }

        #endregion
    }
}
