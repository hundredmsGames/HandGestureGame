using System;
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

        private static int l1_kernel_size = 10;
        private static int l1_stride = 2;

        private static int l2_kernel_size = 3;
        private static int l2_stride = 2;

        private static int fcnn_hidden_neurons = 20;
        private static int fcnn_output_neurons = 10;

        private double learning_rate = 0.01;

        #endregion

        #region Variables

        // Input (Array of Matrix)
        private Matrix input;

        // Weights of layer1 (filters)
        private Matrix l1_kernel;

        private List<Tuple<Location, Location>>[,] l1_kernel_loc_list;

        // Feature Maps of layer1
        private Matrix f_map1;

        // Max Pooling Map of layer2
        private Matrix m_pool1;

        private List<Location> m_pool1_list;

        private Matrix relu1;

        private Matrix cnn_o;

        FCNN fcnn;

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
            l1_kernel = new Matrix(l1_kernel_size, l1_kernel_size);
            l1_kernel.Randomize();

            m_pool1_list = new List<Location>();
            l1_kernel_loc_list = new List<Tuple<Location, Location>>[l1_kernel_size, l1_kernel_size];
            for (int i = 0; i < l1_kernel_size; i++)
            {
                for (int j = 0; j < l1_kernel_size; j++)
                {
                    l1_kernel_loc_list[i, j] = new List<Tuple<Location, Location>>();
                }
            }

            #region This will help while testing

            //Random r = new Random(12312324);

            //this.input = new Matrix(8, 8);
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
            cnn_o = FeedForward(_input);

            Matrix inputDataforFCNN = Matrix.DecreaseToOneDimension(cnn_o);

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
            cnno_d_E = Matrix.IncreaseToTwoDimension(cnno_d_E, cnn_o.rows, cnn_o.cols);

            //Console.WriteLine("\ncnno_d_E\n");
            //Console.WriteLine(cnno_d_E.ToString());

            BackPropagation(cnno_d_E);
        }

        public Matrix Predict(Matrix _input)
        {
            cnn_o = FeedForward(_input);

            Matrix inputDataforFCNN = Matrix.DecreaseToOneDimension(cnn_o);

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

            f_map1 = new Matrix((r_w - f + 2 * p) / s + 1, (c_w - f + 2 * p) / s + 1);

            Convolve(this.input, f_map1, l1_kernel, null, DotProduct, l1_kernel_size, l1_stride, l1_kernel_loc_list);

            //Console.WriteLine("Input\n");
            //Console.WriteLine(this.input.ToString());
            //Console.WriteLine("Kernel1\n");
            //Console.WriteLine(l1_kernel.ToString());
            //Console.WriteLine("FeatureMap1\n");
            //Console.WriteLine(f_map1.ToString());

            //for (int i = 0; i < l1_kernel_size; i++)
            //{
            //    for (int j = 0; j < l1_kernel_size; j++)
            //    {
            //        if (i == 0 && j == 0)
            //        {
            //            Console.WriteLine("Kernel[{0}, {1}]", i, j);
            //            Console.WriteLine(l1_kernel_loc_list[i, j].Count);
            //            foreach (Tuple<Location, Location> t in l1_kernel_loc_list[i, j])
            //            {
            //                Console.WriteLine("{0}, {1}  -  {2}, {3}", t.Item1.r, t.Item1.c, t.Item2.r, t.Item2.c);
            //            }
            //        }
            //    }
            //}

            r_w = f_map1.rows;
            c_w = f_map1.cols;
            f = l2_kernel_size;
            p = 0;
            s = l2_stride;

            m_pool1 = new Matrix((r_w - f + 2 * p) / s + 1, (c_w - f + 2 * p) / s + 1);

            Convolve(f_map1, m_pool1, null, m_pool1_list, MaxPooling, l2_kernel_size, l2_stride);

            //Console.WriteLine("\nMax Pooling1\n");
            //Console.WriteLine(m_pool1.ToString());

            //Console.WriteLine("\nMax Pooling1 List\n");
            //foreach (Location l in m_pool1_list)
            //    Console.WriteLine(l.r + ", " + l.c);

            relu1 = new Matrix(m_pool1.rows, m_pool1.cols);
            relu1 = Matrix.Map(m_pool1, ReLu);

            //Console.WriteLine("\nReLu1\n");
            //Console.WriteLine(relu1.ToString());

            return relu1;
        }

        private void BackPropagation(Matrix cnno_d_E)
        {
            Matrix m_pool1_d_cnno = Matrix.Map(m_pool1, DerOfReLu);

            //Console.WriteLine("\nm_pool1_d_cnno\n");
            //Console.WriteLine(m_pool1_d_cnno.ToString());


            Matrix m_pool1_d_E = Matrix.Multiply(cnno_d_E, m_pool1_d_cnno);

            //Console.WriteLine("\nm_pool1_d_E\n");
            //Console.WriteLine(m_pool1_d_E.ToString());


            Matrix f_map1_d_E = DerOfMaxPooling(m_pool1_list, m_pool1_d_E, f_map1.rows, f_map1.cols);

            //Console.WriteLine("\nf_map1_d_E\n");
            //Console.WriteLine(f_map1_d_E.ToString());


            Matrix kernel1_d_E = DerOfConv(l1_kernel_loc_list, input, f_map1_d_E, l1_kernel_size);
            //Console.WriteLine("\nkernel1_d_E\n");
            //Console.WriteLine(kernel1_d_E.ToString());


            l1_kernel = l1_kernel - (learning_rate * kernel1_d_E);
            //Console.WriteLine("\nl1_kernel\n");
            //Console.WriteLine(l1_kernel.ToString());
        }

        #endregion

        #region Helper Methods

        private static void Convolve(Matrix input, Matrix output, Matrix kernel, List<Location> loc_list,
            Func<int, int, Matrix, Matrix, List<Location>, int, int, int, List<Tuple<Location, Location>>[,], double> func, int kernel_size,
                                     int stride, List<Tuple<Location, Location>>[,] kernel_loc_list = null)
        {
            for (int i = 0, r = 0; r < output.rows && i < input.rows; i += stride, r++)
            {
                for (int j = 0, c = 0; c < output.cols && j < input.cols; j += stride, c++)
                {
                    output[r, c] = func(r, c, input, kernel, loc_list, kernel_size, i, j, kernel_loc_list);
                }
            }
        }

        // FIXME : We use list for location_list but we can use arrays too.
        // using arrays would be a better solution
        // There is no kernel in max pooling so kernel is null.
        private static double MaxPooling(int out_r, int out_c, Matrix input, Matrix kernel,
            List<Location> loc_list, int kernel_size, int rows, int cols, List<Tuple<Location, Location>>[,] kernel_loc_list = null)
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
            List<Location> loc_list, int kernel_size, int rows, int cols, List<Tuple<Location, Location>>[,] kernel_loc_list = null)
        {
            double sum = 0.0f;

            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    sum += kernel[i, j] * input[i + rows, j + cols];

                    Tuple<Location, Location> loc = new Tuple<Location, Location>(
                        new Location(i + rows, j + cols),
                        new Location(out_r, out_c)
                    );

                    kernel_loc_list[i, j].Add(loc);
                }
            }

            return sum;
        }


        private static Matrix DerOfConv(List<Tuple<Location, Location>>[,] loc_list, Matrix curr_layer, Matrix next_layer_d_E,
            int kernel_size)
        {
            Matrix kernel_d_E = new Matrix(kernel_size, kernel_size);

            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    foreach (Tuple<Location, Location> loc in loc_list[i, j])
                    {
                        kernel_d_E[i, j] += curr_layer[loc.Item1.r, loc.Item1.c] * next_layer_d_E[loc.Item2.r, loc.Item2.c];
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