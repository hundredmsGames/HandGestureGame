using System;
using System.Collections.Generic;
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

    partial class CNN
    {
        #region Configuration

        // This should be in a file (txt or json) in future.
        // For simplicity I'll make them static int.

        private int l1_kernel_size = 5;
        private int l1_stride = 2;

        private int l2_kernel_size = 2;
        private int l2_stride = 2;

        private int fcnn_hidden_neurons = 100;
        private int fcnn_output_neurons = 10;

        private double learning_rate = 0.04;

        Func<double, double> activation;
        Func<double, double> derOfActivation;

        #endregion

        #region Variables

        // Input Matrix
        private Matrix input;

        // Kernel (filter) of layer1
        private Matrix l1_kernel;

        // Feature Map of layer1
        private Matrix l1_fmap;

        // Max Pooling Map of layer2
        private Matrix l2_mpool;

        private List<Location> l2_mpool_list;

        private Matrix l3_relu;

        private Matrix cnn_out;

        FCNN fcnn;

        #endregion

        #region Constructors

        public CNN()
        {
            // Randomize weights
            l1_kernel = new Matrix(l1_kernel_size, l1_kernel_size);
            l1_kernel.Randomize();

            activation = ReLu;
            derOfActivation = DerOfReLu;

            l2_mpool_list = new List<Location>();

            // We are deserializing config file here
            Deserialize();

            #region This will help while testing

            //Random r = new Random(12312324);

            //this.input = new Matrix(28, 28);
            //for (int i = 0; i < this.input.rows; i++)
            //{
            //    for (int j = 0; j < this.input.cols; j++)
            //    {
            //        this.input[i, j] = r.NextDouble() * 2f - 1f;
            //    }
            //}

            //l1_kernel = new Matrix(l1_kernel_size, l1_kernel_size);
            //for (int i = 0; i < l1_kernel.rows; i++)
            //{
            //    for (int j = 0; j < l1_kernel.cols; j++)
            //    {
            //        l1_kernel[i, j] = r.NextDouble() * 2f - 1f;
            //    }
            //}

            #endregion
        }

        #endregion

        #region Training

        public void Train(Matrix _input, Matrix _target)
        {
            cnn_out = FeedForward(_input);

            Matrix inputDataforFCNN = Matrix.DecreaseToOneDimension(cnn_out);

            // writing input data to console
            //Console.WriteLine("\nFCNN Input\n");
            //Console.WriteLine(inputDataforFCNN.ToString());

            if (fcnn == null)
            {
                fcnn = new FCNN(
                    inputDataforFCNN.rows * inputDataforFCNN.cols,
                    fcnn_hidden_neurons,
                    fcnn_output_neurons,
                    learning_rate,
                    FCNN.Sigmoid,
                    FCNN.DerSigmoid
                );
            }

            Matrix cnno_d_E = fcnn.Train(inputDataforFCNN, _target);
            cnno_d_E = Matrix.IncreaseToTwoDimension(cnno_d_E, cnn_out.rows, cnn_out.cols);

            //Console.WriteLine("\ncnno_d_E\n");
            //Console.WriteLine(cnno_d_E.ToString());

            BackPropagation(cnno_d_E);
        }

        public Matrix Predict(Matrix _input)
        {
            cnn_out = FeedForward(_input);

            Matrix inputDataforFCNN = Matrix.DecreaseToOneDimension(cnn_out);

            // writing input data to console
            //Console.WriteLine("\nFCNN Input\n");
            //Console.WriteLine(inputDataforFCNN.ToString());

            if (fcnn == null)
            {
                fcnn = new FCNN(
                    inputDataforFCNN.rows * inputDataforFCNN.cols,
                    fcnn_hidden_neurons,
                    fcnn_output_neurons,
                    learning_rate,
                    FCNN.Sigmoid,
                    FCNN.DerSigmoid
                );
            }

            return fcnn.FeedForward(inputDataforFCNN);
        }

        private Matrix FeedForward(Matrix _input)
        {
            this.input = _input;

            // (W−F+2P)/S+1
            int r_w = this.input.rows;
            int c_w = this.input.cols;
            int f = l1_kernel_size;
            int p = 0;
            int s = l1_stride;

            l1_fmap = new Matrix((r_w - f + 2 * p) / s + 1, (c_w - f + 2 * p) / s + 1);

            Convolve(this.input, l1_fmap, l1_kernel, null, DotProduct, l1_kernel_size, l1_stride);

            //Console.WriteLine("Input\n");
            //Console.WriteLine(this.input.ToString());
            //Console.WriteLine("Kernel1\n");
            //Console.WriteLine(l1_kernel.ToString());
            //Console.WriteLine("FeatureMap1\n");
            //Console.WriteLine(f_map1.ToString());

            r_w = l1_fmap.rows;
            c_w = l1_fmap.cols;
            f = l2_kernel_size;
            p = 0;
            s = l2_stride;

            l2_mpool = new Matrix((r_w - f + 2 * p) / s + 1, (c_w - f + 2 * p) / s + 1);

            Convolve(l1_fmap, l2_mpool, null, l2_mpool_list, MaxPooling, l2_kernel_size, l2_stride);

            //Console.WriteLine("\nMax Pooling1\n");
            //Console.WriteLine(m_pool1.ToString());

            //Console.WriteLine("\nMax Pooling1 List\n");
            //foreach (Location l in m_pool1_list)
            //    Console.WriteLine(l.r + ", " + l.c);

            l3_relu = new Matrix(l2_mpool.rows, l2_mpool.cols);
            l3_relu = Matrix.Map(l2_mpool, activation);

            //Console.WriteLine("\nReLu1\n");
            //Console.WriteLine(relu1.ToString());

            return l3_relu;
        }

        private void BackPropagation(Matrix cnno_d_E)
        {
            Matrix l2_mpool_d_cnno = Matrix.Map(l2_mpool, derOfActivation);

            //Console.WriteLine("\nm_pool1_d_cnno\n");
            //Console.WriteLine(m_pool1_d_cnno.ToString());


            Matrix l2_mpool_d_E = Matrix.Multiply(cnno_d_E, l2_mpool_d_cnno);

            //Console.WriteLine("\nm_pool1_d_E\n");
            //Console.WriteLine(m_pool1_d_E.ToString());


            Matrix l1_fmap_d_E = DerOfMaxPooling(l2_mpool_list, l2_mpool_d_E, l1_fmap.rows, l1_fmap.cols);

            //Console.WriteLine("\nf_map1_d_E\n");
            //Console.WriteLine(f_map1_d_E.ToString());


            Matrix l1_kernel_d_E = DerOfConv(input, l1_fmap_d_E, l1_kernel_size, l1_stride);

            //Console.WriteLine("\nkernel1_d_E\n");
            //Console.WriteLine(kernel1_d_E.ToString());


            l1_kernel = l1_kernel - (learning_rate * l1_kernel_d_E);
            //Console.WriteLine("\nl1_kernel\n");
            //Console.WriteLine(l1_kernel.ToString());
        }

        #endregion

        #region Helper Methods

        private static void Convolve(Matrix input, Matrix output, Matrix kernel, List<Location> loc_list,
            Func<int, int, Matrix, Matrix, List<Location>, int, int, int, double> func, int kernel_size,
                                     int stride)
        {
            for (int i = 0, r = 0; r < output.rows && i < input.rows; i += stride, r++)
            {
                for (int j = 0, c = 0; c < output.cols && j < input.cols; j += stride, c++)
                {
                    output[r, c] = func(r, c, input, kernel, loc_list, kernel_size, i, j);
                }
            }
        }

        // FIXME : We use list for location_list but we can use arrays too.
        // using arrays would be a better solution
        // There is no kernel in max pooling so kernel is null.
        private static double MaxPooling(int out_r, int out_c, Matrix input, Matrix kernel,
            List<Location> loc_list, int kernel_size, int rows, int cols)
        {
            double max = double.MinValue;
            int r = -1, c = -1;

            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    if (input[i + rows, j + cols] > max)
                    {
                        max = input[i + rows, j + cols];
                        r = i + rows;
                        c = j + cols;
                    }
                }
            }

            loc_list.Add(new Location(r, c));

            return max;
        }

        // rows and cols are sizes of previous layer, for example: if layer is convolution,
        // we need the size of the corresponding feature map.
        private Matrix DerOfMaxPooling(List<Location> loc_list, Matrix m_pool, int rows, int cols)
        {
            // we need a new matrix that has the same rows and cols with f_map1
            // and filled by zeros
            Matrix prev_layer_d_E = new Matrix(rows, cols);

            //in the list of locations
            int k = 0;

            Location location = loc_list[k];
            for (int i = 0; i < m_pool.rows; i++)
            {
                for (int j = 0; j < m_pool.cols; j++)
                {
                    prev_layer_d_E[location.r, location.c] = m_pool[i, j];
                    k++;
                    if (k < loc_list.Count)
                        location = loc_list[k];
                }
            }

            return prev_layer_d_E;
        }

        private static double DotProduct(int out_r, int out_c, Matrix input, Matrix kernel,
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

        private static Matrix DerOfConv(Matrix input, Matrix output_d_E, int kernel_size, int stride)
        {
            Matrix kernel_d_E = new Matrix(kernel_size, kernel_size);

            // i and j are inputs' indexes
            // r and c are output_d_Es' indexes
            for (int i = 0, r = 0; r < output_d_E.rows && i < input.rows; i += stride, r++)
            {
                for (int j = 0, c = 0; c < output_d_E.cols && j < input.cols; j += stride, c++)
                {
                    for(int p = 0; p < kernel_size; p++)
                    {
                        for(int q = 0; q < kernel_size; q++)
                        {
                            kernel_d_E[p, q] += output_d_E[r, c] * input[i + p, j + q];
                        }
                    }
                }
            }

            return kernel_d_E;
        }

        #endregion

        #region Activation Function (ReLu)
        
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