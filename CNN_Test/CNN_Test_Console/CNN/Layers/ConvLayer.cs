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

            int r_w = this.Input.rows;
            int c_w = this.Input.cols;
            int f = kernel_size;
            int p = 0;
            int s = stride;
            
            double sum = 0;
            int fmapX = 0, fmapY = 0;
            double[,] f_map = new double[(r_w - f + 2 * p) / s + 1, (c_w - f + 2 * p) / s + 1];

            for (int kernel_ID = 0; kernel_ID < kernels.Length; kernel_ID++)
            {
                for (int x = 0; x < Input.rows; x += stride)
                {
                    for (int y = 0; y < Input.cols; y += stride)
                    {
                        sum = 0;
                        for (int i = 0; i < kernel_size; i++)
                        {
                            for (int j = 0; j < kernel_size; j++)
                            {
                                sum += Input[x, y] * kernels[kernel_ID][i, j];
                            }
                        }
                    }
                    f_map[fmapX, fmapY] = sum;
                    //now we are moving next cell in current row
                    fmapX++;
                }
                fmapX = 0;
                //move to next row
                fmapY++;

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
