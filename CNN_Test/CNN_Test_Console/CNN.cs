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

        private Matrix relu1;

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

            Convolve(this.input, f_map1, l1_kernel, DotProduct, l1_kernel_size, l1_stride);

            Console.WriteLine("Convolution\n");
            Console.WriteLine(this.input.ToString());
            Console.WriteLine(l1_kernel.ToString());
            Console.WriteLine(f_map1.ToString());

            r_w = f_map1.rows;
            c_w = f_map1.cols;
            f = l2_kernel_size;
            p = 0;
            s = l2_stride;

            m_pool1 = new Matrix((r_w - f + 2 * p) / s + 1, (c_w - f + 2 * p) / s + 1);

            Convolve(f_map1, m_pool1, null, MaxPooling, l2_kernel_size, l2_stride);

            Console.WriteLine("\nMax Pooling\n");
            Console.WriteLine(m_pool1.ToString());

            relu1 = new Matrix(m_pool1.rows, m_pool1.cols);
            relu1 = Matrix.Map(m_pool1, ReLu);

            Console.WriteLine("\nReLu\n");
            Console.WriteLine(relu1.ToString());
        }

        private void BackPropagation()
        {

        }



        private static void Convolve(Matrix input, Matrix output, Matrix kernel,
            Func<Matrix, Matrix, int, int, int, double> func, int kernel_size, int stride)
        {
            for (int i = 0, r = 0; r < output.rows && i < input.rows; i += stride, r++)
            {
                for (int j = 0, c = 0; c < output.cols && j < input.cols; j += stride, c++)
                {
                    output[r, c] = func(input, kernel, kernel_size, i, j);
                }
            }
        }

        // There is no kernel in max pooling so kernel is null.
        private static double MaxPooling(Matrix input, Matrix kernel, int kernel_size, int rows, int cols)
        {
            double max = double.MinValue;

            for(int i = 0; i < kernel_size; i++)
            {
                for(int j = 0; j < kernel_size; j++)
                {
                    max = Math.Max(max, input[i + rows, j + cols]);
                }
            }

            return max;
        }

        private static double DotProduct(Matrix input, Matrix kernel, int kernel_size, int rows, int cols)
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

        #endregion
    }
}