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
