using System;
using MatrixLib;

namespace ConvNeuralNetwork
{
    class MaxPoolingLayer : Layer
    {
        #region Variables

        private int kernel_size;
        private int stride;

        // Location of maximum values in output for each channel
        private Location[,,] max_locations;

        #endregion

        #region Constructors

        public MaxPoolingLayer(int kernel_size, int stride) : base(LayerType.MAXPOOLING)
        {
            this.Kernel_Size = kernel_size;
            this.Stride = stride;
        }

        public override void Initialize()
        {
            base.Initialize();

            Input = InputLayer.Output;
            Output = new Matrix[Input.Length];
            Output_d_E = new Matrix[Input.Length];

            int in_r = this.Input[0].rows;
            int in_c = this.Input[0].cols;
            int f = Kernel_Size;
            int p = 0;
            int s = Stride;

            int out_size_r = (in_r - f + 2 * p) / s + 1;
            int out_size_c = (in_c - f + 2 * p) / s + 1;


            // Initialize max_locations
            this.max_locations = new Location[this.Input.Length, out_size_r, out_size_c];

            for (int i = 0; i < this.Input.Length; i++)
            {
                // Initialize output
                this.Output[i] = new Matrix(out_size_r, out_size_c);

                // Initialize output_d_E
                this.Output_d_E[i] = new Matrix(out_size_r, out_size_c);
            } 
        }
        
        #endregion

        #region Methods

        public override void FeedForward()
        {
            base.FeedForward();

            int out_idx_r, out_idx_c;

            for (int ch = 0; ch < Input.Length; ch++)
            {
                out_idx_r = 0;
                for (int r = 0; r < Input[0].rows && out_idx_r < Output[0].rows; r += stride, out_idx_r++)
                {
                    out_idx_c = 0;
                    for (int c = 0; c < Input[0].cols && out_idx_c < Output[0].cols; c += stride, out_idx_c++)
                    {
                        float max = float.MinValue;
                        int max_r = 0, max_c = 0;

                        for (int i = 0; i < this.Kernel_Size; i++)
                        {
                            for (int j = 0; j < this.Kernel_Size; j++)
                            {
                                if (Input[ch][i + r, j + c] > max)
                                {
                                    max = Input[ch][i + r, j + c];
                                    max_r = i + r;
                                    max_c = j + c;
                                }
                            }
                        }

                        max_locations[ch, out_idx_r, out_idx_c] = new Location(max_r, max_c);
                        Output[ch][out_idx_r, out_idx_c] = max;
                    }             
                }
            }
            
            this.OutputLayer.Input = Output;
        }

        public override void Backpropagation()
        {
            base.Backpropagation();

            for (int ch = 0; ch < Input.Length; ch++)
            {
                // reset gradients
                InputLayer.Output_d_E[ch].FillZero();

                for (int i = 0; i < Output[ch].rows; i++)
                {
                    for (int j = 0; j < Output[ch].cols; j++)
                    {
                        Location max = max_locations[ch, i, j];
                        InputLayer.Output_d_E[ch][max.r, max.c] = Output_d_E[ch][i, j];
                    }
                }
            }
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
