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
            
            
        }

        #endregion

        #region Methods
        public override void Initialize()
        {
            base.Initialize();
            // Initialize output
            this.Input = InputLayer.Output;

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
            //for every filter we need to do this
            Matrix output_d_E = this.Output_d_E[0];

            Matrix kernel_d_E = new Matrix(kernel_size, kernel_size);
            for (int ch = 0; ch < Filters; ch++)
            {
                for (int i = 0, r = 0; r < output_d_E.rows && i < Input[ch].rows; i += stride, r++)
                {
                    for (int j = 0, c = 0; c < output_d_E.cols && j < Input[ch].cols; j += stride, c++)
                    {
                        for (int p = 0; p < kernel_size; p++)
                        {
                            for (int q = 0; q < kernel_size; q++)
                            {
                                kernel_d_E[p, q] += output_d_E[r, c] * Input[ch][i + p, j + q];

                                if (LayerIndex != 0)
                                    this.InputLayer.Output_d_E[ch][i + p, j + q] += kernels[ch][p, q] * output_d_E[r, c];
                            }
                        }
                    }
                }
                this.kernels[ch] = this.kernels[ch] - (Network.LearningRate * kernel_d_E);
            }
            

        }
        private static Matrix DerOfConv(Matrix input, Matrix output_d_E, int kernel_size, int stride, Matrix kernel = null, Matrix input_d_E = null)
        {
            Matrix kernel_d_E = new Matrix(kernel_size, kernel_size);

            // i and j are inputs' indexes
            // r and c are output_d_Es' indexes
            for (int i = 0, r = 0; r < output_d_E.rows && i < input.rows; i += stride, r++)
            {
                for (int j = 0, c = 0; c < output_d_E.cols && j < input.cols; j += stride, c++)
                {
                    for (int p = 0; p < kernel_size; p++)
                    {
                        for (int q = 0; q < kernel_size; q++)
                        {
                            kernel_d_E[p, q] += output_d_E[r, c] * input[i + p, j + q];

                            if (input_d_E != null)
                                input_d_E[i + p, j + q] += kernel[p, q] * output_d_E[r, c];
                        }
                    }
                }
            }

            return kernel_d_E;
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
