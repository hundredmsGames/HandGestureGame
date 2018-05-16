using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MatrixLib;

namespace ConvNeuralNetwork
{
    class InputLayer : Layer
    {
        #region Variables

        private int width;
        private int height;
        private int channels;

        #endregion

        #region Constructors

        public InputLayer(int width, int height, int channels) : base(LayerType.INPUT)
        {
            this.width    = width;
            this.height   = height;
            this.channels = channels;
        }

        #endregion

        #region Methods

        public override void FeedForward(Matrix[] input)
        {
            base.FeedForward(input);

            //DO WHAT YOU NEED TO DO
            OutputLayer.Input = input;
        }

        #endregion

        #region Properties

        public int Width { get => width; set => width = value; }

        public int Height { get => height; set => height = value; }

        public int Channels { get => channels; set => channels = value; }

        #endregion
    }
}
