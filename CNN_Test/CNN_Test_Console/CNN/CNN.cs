using System;
using System.Collections.Generic;
using MatrixLib;

namespace ConvNeuralNetwork
{
    partial class CNN
    {
        #region Variables

        private Layer[] layers;

        private int nextLayerIndex;

        private float learningRate;

        private Matrix target;

        #endregion

        #region Constructors

        public CNN()
        {
            // We are deserializing config file at the top of the constructor
            Description[] descriptions = DeserializeConfig();
            layers = new Layer[descriptions.Length];

            for (int i = 0; i < descriptions.Length; i++)
            {
                NewLayer(descriptions[i]);
                Console.WriteLine(descriptions[i].ToString());
            }

            // First layer index is 0.
            nextLayerIndex = 0;


        }

        #endregion

        #region Methods

        public void NewLayer(Description description)
        {
            Layer newLayer = null;

            switch (description.layerType)
            {
                case LayerType.INPUT:
                    // FIXME: We have a problem here. Probably we need to hold input array in description
                    newLayer = new InputLayer(description.width, description.height, description.channels);
                    break;

                case LayerType.CONVOLUTIONAL:
                    newLayer = new ConvLayer(description.filters, description.kernelSize, description.stride, description.padding);
                    break;

                case LayerType.MAXPOOLING:
                    newLayer = new MaxPoolingLayer(description.kernelSize, description.stride);
                    break;

                case LayerType.FULLY_CONNECTED:
                    //lenght-2 because lenght-1 is FCNN layer so we are getting null ref exception
                    Layer previousLayer = layers[layers.Length - 2];
                    int inputNeurons = previousLayer.Output.Length * previousLayer.Output[0].cols * previousLayer.Output[0].rows;
                    //FIXME:think about topology and find a better way to handle it
                    newLayer = new FullyConLayer(new int[] { inputNeurons, description.hiddens, description.outputs }, description.activationHidden, description.activation);
                    break;

                default:
                    throw new UndefinedLayerException("This is not a recognizeable LayerType " + description.layerType.ToString() + " You might be writing it wrong to config file");
            }
            //every layer knows the CNN ref
            newLayer.Network = this;

            if (nextLayerIndex == 0)
            {
                //if this is the first layer and this is not an input layer so we have a problem
                if (description.layerType != LayerType.INPUT)
                    throw new WrongLayerException("You have to start with Input Layer");
            }
            else
            {
                Layer prevLayer = layers[this.nextLayerIndex - 1];
                prevLayer.OutputLayer = newLayer;
                newLayer.InputLayer = prevLayer;

            }

            newLayer.Initialize();

            newLayer.LayerIndex = this.nextLayerIndex;
            this.layers[this.nextLayerIndex] = newLayer;
            this.nextLayerIndex++;
        }

        #endregion

        #region Properties

        public Layer[] Layers
        {
            get { return layers; }
            set { layers = value; }
        }

        public int NextLayerIndex
        {
            get { return nextLayerIndex; }
            set { nextLayerIndex = value; }
        }

        public float LearningRate
        {
            get { return learningRate; }
            set { learningRate = value; }
        }

        public Matrix Target
        {
            get { return target; }
            set { target = value; }
        }

        #endregion

        #region OLD METHODS (WILL BE DELETED)

        private static void Convolve(Matrix input, Matrix output, Matrix kernel, List<Location> loc_list,
            Func<int, int, Matrix, Matrix, List<Location>, int, int, int, float> func, int kernel_size,
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

        private static Matrix DerOfConv(Matrix input, Matrix output_d_E, int kernel_size, int stride, Matrix kernel = null, Matrix input_d_E = null)
        {
            Matrix kernel_d_E = new Matrix(kernel_size, kernel_size);

            // i and j are inputs' indexes
            // r and c are output_d_Es' indexes
            for (int i = 0, r = 0; r < output_d_E.rows && i < input.rows; i += stride, r++)
            {
                for (int j = 0, c = 0; c < output_d_E.cols && j < input.cols; j += stride, c++)
                {
                    for (int p = 0; p < kernel_size; p++)
                    {
                        for (int q = 0; q < kernel_size; q++)
                        {
                            kernel_d_E[p, q] += output_d_E[r, c] * input[i + p, j + q];

                            if (input_d_E != null)
                                input_d_E[i + p, j + q] += kernel[p, q] * output_d_E[r, c];
                        }
                    }
                }
            }

            return kernel_d_E;
        }

        #endregion
    }
}