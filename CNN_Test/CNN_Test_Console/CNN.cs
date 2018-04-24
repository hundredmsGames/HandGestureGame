﻿using System;
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

        private static int l1_kernel_size = 3;
        private static int l1_stride = 1;

        private static int l2_kernel_size = 2;
        private static int l2_stride = 2;

        #endregion

        #region Variables

        // Input (Array of Matrix)
        private Matrix input;

        // Weights of layer1 (filters)
        private Matrix l1_w;

        // Feature Maps of layer1
        private Matrix f_map1;

        // Max Pooling Map of layer2
        private Matrix m_pool1;

        #endregion

        /* Helper Example
         * 
         * Input:        8 x 8
         * L1_Filter:    3 x 3
         * Feature_Map:  6 x 6
         * 
         * 
         * 
         */

        #region Constructors

        public CNN()
        {
            //// Randomize weights
            //l1_w = new Matrix(l1_filter_rows, l1_filter_cols);
            //l1_w.Randomize();

            #region This will help while testing

            Random r = new Random(1231234124);

            this.input = new Matrix(8, 8);
            for (int i = 0; i < this.input.rows; i++)
            {
                for (int j = 0; j < this.input.cols; j++)
                {
                    this.input[i, j] = r.NextDouble();
                }
            }

            l1_w = new Matrix(l1_kernel_size, l1_kernel_size);
            for (int i = 0; i < l1_w.rows; i++)
            {
                for (int j = 0; j < l1_w.cols; j++)
                {
                    l1_w[i, j] = r.NextDouble();
                }
            }

            #endregion
        }

        #endregion

        #region Methods

        public void Train()
        {
            FeedForward(null);
        }

        private void FeedForward(Matrix input)
        {
            // TESTING
            //this.input = input;

            // (W−F+2P)/S+1
            int r_w = this.input.rows;
            int c_w = this.input.cols;
            int f = l1_kernel_size;
            int p = 0;
            int s = l1_stride;

            f_map1 = new Matrix((r_w - f + 2 * p) / s + 1, (c_w - f + 2 * p) / s + 1);

            int f_i = 0, f_j = 0;
            for (int i = 0; f_i < f_map1.rows && i < this.input.rows; i += s)
            {
                for (int j = 0; f_j < f_map1.cols && j < this.input.cols; j += s)
                {
                    f_map1[f_i, f_j] = Matrix.DotProduct(this.input, l1_w, i, j);
                    f_j++;
                }
                f_j = 0;
                f_i++;
            }

            Console.WriteLine(this.input.ToString());
            Console.WriteLine(l1_w.ToString());
            Console.WriteLine(f_map1.ToString());

            r_w = f_map1.rows;
            c_w = f_map1.cols;
            f = l2_kernel_size;
            p = 0;
            s = l2_stride;


            m_pool1 = MaxPooling(f_map1, f, s);
            //new Matrix((r_w - f + 2 * p) / s + 1, (c_w - f + 2 * p) / s + 1);



        }

        //matrix, row_start, col_start, size
        private Matrix MaxPooling(Matrix m, int size, int stride)
        {
            double max = 0;
            int r_w = m.rows;
            int c_w = m.cols;
            int p_r = 0, p_c = 0;
            int p = 0;
            int f = l2_kernel_size;
            Matrix _matrix = new Matrix((r_w - f + 2 * p) / stride + 1, (c_w - f + 2 * p) / stride + 1);
            for (int k = 0; k < m.rows; k += stride)
            {
                for (int l = 0; l < m.cols; l += stride)
                {
                    for (int i = k; i < k + size; i++)
                    {
                        for (int j = l; j < l + size; j++)
                        {
                            if (m[i, j] > max)
                            {
                                max = m[i, j];
                            }
                        }
                    }
                    //pull max and push to the matrix
                    _matrix[p_r, p_c] = max;

                    p_c++;
                }
                p_r++;
            }
            return _matrix;
        }
        private void BackPropagation()
        {

        }
    }
}

        #endregion
