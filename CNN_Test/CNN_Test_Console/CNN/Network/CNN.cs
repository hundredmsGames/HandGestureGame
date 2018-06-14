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

        private double learningRate;

        private Matrix target;

        Description[] descriptions;

        #endregion

        #region Constructors

        public CNN(string fileName=null)
        {
            // We are deserializing config file at the top of the constructor
            if (string.IsNullOrEmpty(fileName))
            {
                descriptions = DeserializeConfig();
                layers = new Layer[descriptions.Length];

                for (int i = 0; i < descriptions.Length; i++)
                {
                    NewLayer(descriptions[i]);
                }
            }
            else
            {
                //load descriptions from saved file
                LoadData(fileName);
            }

            // First layer index is 0.
            //THINK: What do we do with this line of code????
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
                    newLayer = new InputLayer(description.width, description.height, description.channels);
                    break;

                case LayerType.CONVOLUTIONAL:
                    newLayer = new ConvLayer(description.filters, description.kernelSize, description.stride, description.activation, description.padding);
                    break;

                case LayerType.MAXPOOLING:
                    newLayer = new MaxPoolingLayer(description.kernelSize, description.stride);
                    break;

                case LayerType.FULLY_CONNECTED:

                    // TODO: This code should be arranged with respect to new FC Layer Model

                    Layer previousLayer = layers[nextLayerIndex - 1];
                    int[] topology = new int[description.layers + 1];

                    topology[0] = previousLayer.Output.Length * previousLayer.Output[0].cols * previousLayer.Output[0].rows;
                    for (int i = 1; i < topology.Length; i++)
                        topology[i] = description.neurons[i - 1];

                    //FIXME:think about topology and find a better way to handle it
                    newLayer = new FC_Network(topology, description.fc_activations.ToArray());
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

        public Matrix Predict(Matrix[] _input)
        {
            layers[0].FeedForward(_input);
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].FeedForward();
            }

            // Output[0] is a little bit hardcoded. Find better way later.
            return layers[layers.Length - 1].Output[0];
        }

        public void Train(Matrix[] _input, Matrix _target)
        {
            target = _target;

            Predict(_input);

            for (int i = layers.Length - 1; i >= 0; i--)
            {
                layers[i].Backpropagation();
            }
        }

        public double GetError()
        {
            // Calculate the error 
            // ERROR = (1 / 2) * (TARGETS - OUTPUTS)^2

            //double error = 0f;
            //if (target != null)
            //{
            //    Matrix outputError = target - Layers[Layers.Length - 1].Output[0];
            //    outputError = Matrix.Multiply(outputError, outputError) / 2f;

            //    for (int i = 0; i < outputError.rows; i++)
            //        error += outputError[i, 0];
            //}

            //return error;

            return CrossEntropy(Layers[Layers.Length - 1].Output[0], target);
        }

        public double CrossEntropy(Matrix p, Matrix c)
        {
            if (target == null)
                return -1.0;

            double error = 0.0;
            Matrix errorMatrix = -Matrix.Multiply(c, Matrix.Log(p)) + Matrix.Multiply(1 - c, Matrix.Log(1 - p));

            for (int i = 0; i < errorMatrix.rows; i++)
                error += errorMatrix[i, 0];

            return error;
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

        public double LearningRate
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
    }
}