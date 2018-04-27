﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MatrixLib;
using FullyConnectedNN;

namespace ConvNeuralNetwork
{
    struct Location
    {
        public int r;
        public int c;

        public Location(int _r, int _c)
        {
            r = _r;
            c = _c;
        }
    }

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
        private Matrix l1_kernel;

        // Feature Maps of layer1
        private Matrix f_map1;

        // Max Pooling Map of layer2
        private Matrix m_pool1;

        private List<Location> m_pool1_list;

        private Matrix relu1;

        private Matrix cnn_o;

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

            m_pool1_list = new List<Location>();

            #region This will help while testing

            Random r = new Random(12312324);

            this.input = new Matrix(8, 8);
            for (int i = 0; i < this.input.rows; i++)
            {
                for (int j = 0; j < this.input.cols; j++)
                {
                    this.input[i, j] = r.NextDouble() * 2f - 1f;
                }
            }

            l1_kernel = new Matrix(l1_kernel_size, l1_kernel_size);
            for (int i = 0; i < l1_kernel.rows; i++)
            {
                for (int j = 0; j < l1_kernel.cols; j++)
                {
                    l1_kernel[i, j] = r.NextDouble() * 2f - 1f;
                }
            }

            #endregion
        }

        #endregion

        #region Methods

        public void Train(Matrix _input, Matrix _target)
        {
            cnn_o = FeedForward(_input);

            Matrix inputDataforFCNN = Matrix.DecreaseToOneDimension(cnn_o);

            // writing input data to console
            Console.WriteLine("\nFCNN Input\n");
            Console.WriteLine(inputDataforFCNN.ToString());

            FCNN fcnn = new FCNN(
                inputDataforFCNN.rows * inputDataforFCNN.cols,
                3,
                2,
                0.1,
                FCNN.Sigmoid,
                FCNN.DerSigmoid
            );

            Matrix cnno_d_E = fcnn.Train(inputDataforFCNN, _target);
            cnno_d_E = Matrix.IncreaseToTwoDimension(cnno_d_E, cnn_o.rows, cnn_o.cols);

            Console.WriteLine("\ncnno_d_E\n");
            Console.WriteLine(cnno_d_E.ToString());

            BackPropagation(cnno_d_E);
        }

        private Matrix FeedForward(Matrix _input)
        {
            //  T E S T I N G
            //this.input = _input;

            // (W−F+2P)/S+1
            int r_w = this.input.rows;
            int c_w = this.input.cols;
            int f = l1_kernel_size;
            int p = 0;
            int s = l1_stride;

            f_map1 = new Matrix((r_w - f + 2 * p) / s + 1, (c_w - f + 2 * p) / s + 1);

            Convolve(this.input, f_map1, l1_kernel, null, DotProduct, l1_kernel_size, l1_stride);

            Console.WriteLine("Input\n");
            Console.WriteLine(this.input.ToString());
            Console.WriteLine("Kernel1\n");
            Console.WriteLine(l1_kernel.ToString());
            Console.WriteLine("FeatureMap1\n");
            Console.WriteLine(f_map1.ToString());

            r_w = f_map1.rows;
            c_w = f_map1.cols;
            f = l2_kernel_size;
            p = 0;
            s = l2_stride;

            m_pool1 = new Matrix((r_w - f + 2 * p) / s + 1, (c_w - f + 2 * p) / s + 1);

            Convolve(f_map1, m_pool1, null, m_pool1_list, MaxPooling, l2_kernel_size, l2_stride);

            Console.WriteLine("\nMax Pooling1\n");
            Console.WriteLine(m_pool1.ToString());

            Console.WriteLine("\nMax Pooling1 List\n");
            foreach (Location l in m_pool1_list)
                Console.WriteLine(l.r + ", " + l.c);

            relu1 = new Matrix(m_pool1.rows, m_pool1.cols);
            relu1 = Matrix.Map(m_pool1, ReLu);

            Console.WriteLine("\nReLu1\n");
            Console.WriteLine(relu1.ToString());

            return relu1;
        }

        private void BackPropagation(Matrix cnno_d_E)
        {

            // m_pool1__d__E = m_pool1__d__cnno  * in_d_E
            Matrix m_pool1_d_cnno = Matrix.Map(m_pool1, DerOfReLu);

            Console.WriteLine("\nm_pool1_d_cnno\n");
            Console.WriteLine(m_pool1_d_cnno.ToString());

            Matrix m_pool1_d_E = Matrix.Multiply(cnno_d_E, m_pool1_d_cnno);

            Console.WriteLine("\nm_pool1_d_E\n");
            Console.WriteLine(m_pool1_d_E.ToString());


            // f_map1__d__E = f_map1__d__m_pool1 * m_pool1__d__E
        }



        private static void Convolve(Matrix input, Matrix output, Matrix kernel, List<Location> loc_list,
            Func<Matrix, Matrix, List<Location>, int, int, int, double> func, int kernel_size, int stride)
        {
            for (int i = 0, r = 0; r < output.rows && i < input.rows; i += stride, r++)
            {
                for (int j = 0, c = 0; c < output.cols && j < input.cols; j += stride, c++)
                {
                    output[r, c] = func(input, kernel, loc_list, kernel_size, i, j);
                }
            }
        }

        //FIXME : We use list for location_list but we can use arrays too.
        //using arrays would be a better solution
        // There is no kernel in max pooling so kernel is null.
        private static double MaxPooling(Matrix input, Matrix kernel,
            List<Location> loc_list, int kernel_size, int rows, int cols)
        {
            double max = double.MinValue;
            int r = -1, c = -1;

            for (int i = 0; i < kernel_size; i++)
            {
                max = double.MinValue;
                for (int j = 0; j < kernel_size; j++)
                {
                    if (input[i + rows, j + cols] > max)
                    {
                        max = input[i + rows, j + cols];
                        r = i + rows;
                        c = j + cols;
                    }
                }

                loc_list.Add(new Location(r, c));
            }

            return max;
        }

        private static double DotProduct(Matrix input, Matrix kernel,
            List<Location> loc_list, int kernel_size, int rows, int cols)
        {
            double sum = 0.0f;

            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    sum += kernel[i, j] * input[i + rows, j + cols];
                }
            }

            return sum;
        }

        private static double ReLu(double x)
        {
            return Math.Max(x, 0);
        }

        private static double DerOfReLu(double x)
        {
            return (x > 0) ? 1.0 : 0.0;
        }

        #endregion
    }
}