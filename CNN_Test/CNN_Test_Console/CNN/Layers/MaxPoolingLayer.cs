using MatrixLib;
using System.Collections.Generic;

namespace ConvNeuralNetwork
{
    class MaxPoolingLayer : Layer
    {
        #region Variables

        private int kernel_size;
        private int stride;
        List<Location>[] loc_list;
        #endregion

        #region Constructors

        public MaxPoolingLayer(int kernel_size, int stride) : base(LayerType.MAXPOOLING)
        {
            this.Kernel_Size = kernel_size;
            this.Stride = stride;

            // Initialize output
            int in_r = this.Input[0].rows;
            int in_c = this.Input[0].cols;
            int f = Kernel_Size;
            int p = 0;
            int s = Stride;

            int out_size_r = (in_r - f + 2 * p) / s + 1;
            int out_size_c = (in_c - f + 2 * p) / s + 1;
            this.Output = new Matrix[this.Input.Length];
            for (int i = 0; i < this.Input.Length; i++)
            {
                this.Output[i] = new Matrix(out_size_r, out_size_c);
            }

            // Initialize output_d_E
            this.Output_d_E = new Matrix[this.Input.Length];
            for (int i = 0; i < this.Input.Length; i++)
            {
                this.Output_d_E[i] = new Matrix(out_size_r, out_size_c);
            }
        }

        #endregion


        #region Methods

        public override void FeedForward()
        {
            base.FeedForward();

            //recreating everytime because we dont want to add more than we need
            loc_list = new List<Location>[Input.Length];
            for (int i = 0; i < loc_list.Length; i++)
            {
                loc_list[i] = new List<Location>();
            }

            for (int index = 0; index < Input.Length; index++)
            {
                int out_row_idx = 0, out_col_idx = 0;

                for (int r = 0; r < Input[0].rows && out_row_idx < Output[0].rows; r += stride, out_row_idx++)
                {
                    for (int c = 0; c < Input[0].cols && out_col_idx < Output[0].cols; c += stride, out_col_idx++)
                    {
                        double max = double.MinValue;
                        for (int i = 0; i < this.Kernel_Size; i++)
                        {
                            for (int j = 0; j < this.Kernel_Size; j++)
                            {
                                if (Input[index][i + r, j + c] > max)
                                {
                                    max = Input[index][i + r, j + c];
                                    r = i + r;
                                    c = j + c;
                                }
                            }
                        }
                        loc_list[index].Add(new Location(r, c));
                        Output[index][out_row_idx, out_col_idx] = max;
                    }             
                }
            }
        }

        public override void Backpropagation()
        {
            base.Backpropagation();

            for (int ch = 0; ch < this.Input.Length; ch++)
            {
                //in the list of locations
                int k = 0;
                Location location = loc_list[ch][k];

                for (int i = 0; i < Output[ch].rows; i++)
                {
                    for (int j = 0; j < Output[ch].cols; j++)
                    {
                        InputLayer.Output_d_E[ch][location.r, location.c] = Output[ch][i, j];
                        k++;

                        if (k < loc_list[ch].Count)
                            location = loc_list[ch][k];
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
