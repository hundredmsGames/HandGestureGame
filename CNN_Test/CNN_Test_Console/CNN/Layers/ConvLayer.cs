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

        public ConvLayer(int filters, int kernel_size, int stride, int padding = -1) : base(LayerType.CONVOLUTIONAL)
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

            // Padding
            if(padding == -1)
            {
                // padding = default value
            }

            // Initialize output
            int in_r = this.Input[0].rows;
            int in_c = this.Input[0].cols;
            int f = Kernel_Size;
            int p = Padding;
            int s = Stride;

            int out_size_r = (in_r - f + 2 * p) / s + 1;
            int out_size_c = (in_c - f + 2 * p) / s + 1;
            this.Output = new Matrix[filters];
            for (int i = 0; i < filters; i++)
            {
                this.Output[i] = new Matrix(out_size_r, out_size_c);
            }
        }

        #endregion

        #region Methods

        public override void FeedForward()
        {
            base.FeedForward();

            int out_idx_r = 0;
            int out_idx_c = 0;

            for (int ch_idx = 0; ch_idx < Filters; ch_idx++)
            {
                for (int r = 0; r < Input[0].rows && out_idx_r < Output[0].rows; r += stride, out_idx_r++)
                {
                    for (int c = 0; c < Input[0].cols && out_idx_c < Output[0].cols; c += stride, out_idx_c++)
                    {
                        for (int i = 0; i < kernel_size; i++)
                        {
                            for (int j = 0; j < kernel_size; j++)
                            {
                                Output[ch_idx][out_idx_r, out_idx_c] += Input[ch_idx][r, c] * kernels[ch_idx][i, j];
                            }
                        }
                    }
                }
            }
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
