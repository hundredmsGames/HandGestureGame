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
                //Console.WriteLine(descriptions[i].ToString());
            }

            // First layer index is 0.
            nextLayerIndex = 0;
        }

        #endregion

        #region Methods

        public void Train(Matrix _input, Matrix _target)
        {
            target = _target;

            Predict(_input);

            for (int i = layers.Length - 1; i >= 0; i--)
            {
                layers[i].Backpropagation();
            }
        }

        public Matrix Predict(Matrix _input)
        {
            layers[0].FeedForward(_input);
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].FeedForward();
            }
            return layers[layers.Length - 1].Output[0];
        }

        public void NewLayer(Description description)
        {
            Layer newLayer = null;

            switch (description.layerType)
            {
                case LayerType.INPUT:
                    newLayer = new InputLayer(description.width, description.height, description.channels);
                    break;

                case LayerType.CONVOLUTIONAL:
                    newLayer = new ConvLayer(description.filters, description.kernelSize, description.stride, description.padding);
                    break;

                case LayerType.MAXPOOLING:
                    newLayer = new MaxPoolingLayer(description.kernelSize, description.stride);
                    break;

                case LayerType.FULLY_CONNECTED:

                    Layer previousLayer = layers[nextLayerIndex - 1];
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
    }
}