using MatrixLib;
using System.Collections.Generic;

namespace ConvNeuralNetwork
{
    class MaxPoolingLayer : Layer
    {
        #region Variables

        private int kernel_size;
        private int stride;
        List<Location> loc_list;
        #endregion

        #region Constructors

        public MaxPoolingLayer(int kernel_size, int stride) : base(LayerType.MAXPOOLING)
        {
            this.Kernel_Size = kernel_size;
            this.Stride = stride;
            
        }

        #endregion


        #region MaxPoolingMethod

        private double MaxPooling()
        {
            double max = double.MinValue;

            //recreating everytime because we dont want to add more than we need
            loc_list = new List<Location>();
            for (int r = 0; r < Input.rows; r+=stride)
            {
                for (int c = 0; c < Input.cols; c+=stride)
                {
                    for (int i = 0; i < this.Kernel_Size; i++)
                    {
                        for (int j = 0; j < this.Kernel_Size; j++)
                        {
                            if (Input[i + r, j + c ] > max)
                            {
                                max = Input[i + r, j + c];
                                r = i + r;
                                c = j + c;
                            }
                        }
                    }
                    
                    loc_list.Add(new Location(r, c));
                } 
            }

            

            return max;
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
