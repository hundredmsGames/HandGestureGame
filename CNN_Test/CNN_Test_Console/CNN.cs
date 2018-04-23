using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MatrixLib;

namespace ConvNeuralNetwork
{
    class CNN
    {
        #region Configuration

        // This should be in a file (txt or json) in future.
        // For simplicity I'll make them static int.

        private static int l1_filter_size = 2;
        private static int l1_filter_rows = 3;
        private static int l1_filter_cols = 3;
        private static int l1_stride = 1;

        #endregion

        #region Variables

        // Input (Array of Matrix)
        private Matrix[] input;

        // Weights of layer1 (filters)
        private Matrix[] l1_w;

        // Feature Maps of layer1
        private Matrix[] f_map1;


        // Size of input
        private int input_size;

        // Size of layer1
        private int l1_size = l1_filter_size;

        #endregion

        /* Helper Example
         * 
         * Input:        3 x 8 x 8
         * L1_Filter:    2 x 3 x 3
         * Feature_Map:  6 x 6 x 6
         * 
         */

        #region Constructors

        public CNN(Matrix[] input)
        {
            this.input = input;
            this.input_size = input.Length;

            //// Randomize weights
            //l1_w = new Matrix[l1_size];
            //for (int i = 0; i < l1_size; i++)
            //{
            //    l1_w[i] = new Matrix(l1_filter_rows, l1_filter_cols);
            //    l1_w[i].Randomize();
            //}

            #region This will help while testing

            l1_w = new Matrix[l1_size];
            l1_w[0] = new Matrix(l1_filter_rows, l1_filter_cols);
            l1_w[1] = new Matrix(l1_filter_rows, l1_filter_cols);

            l1_w[0].data[0, 0] = 0.12f;
            l1_w[0].data[0, 1] = -0.3f;
            l1_w[0].data[0, 2] = -0.9f;
            l1_w[0].data[1, 0] = 0.54f;
            l1_w[0].data[1, 1] = 0.45f;
            l1_w[0].data[1, 2] = 0.99f;
            l1_w[0].data[2, 0] = 0.01f;
            l1_w[0].data[2, 1] = -0.04f;
            l1_w[0].data[2, 2] = -0.4f;

            l1_w[1].data[0, 0] = -0.42f;
            l1_w[1].data[0, 1] = 0.5f;
            l1_w[1].data[0, 2] = -0.1f;
            l1_w[1].data[1, 0] = 0.74f;
            l1_w[1].data[1, 1] = 0.15f;
            l1_w[1].data[1, 2] = -0.19f;
            l1_w[1].data[2, 0] = -0.51f;
            l1_w[1].data[2, 1] = 0.06f;
            l1_w[1].data[2, 2] = 0.3f;

            #endregion
        }

        #endregion

        #region Methods

        public void Train()
        {

        }

        private void FeedForward()
        {
            f_map1 = new Matrix[input_size * l1_size];

            for(int i = 0; i < input_size; i++)
            {
                for(int j = 0; j < l1_size; j++)
                {
                    
                }
            }
        }

        private void BackPropagation()
        {

        }

        #endregion
    }
}
