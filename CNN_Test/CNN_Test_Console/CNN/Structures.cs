using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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

    struct Description
    {
        // Network Description
        public float learningRate;

        // input desc
        public int width;
        public int height;
        public int channels;

        // conv & pool desc
        public int filters;
        public int kernelSize;
        public int stride;
        public int padding;
        public ActivationType activation;

        // fc layer
        public int hiddens;
        public int outputs;

        public LayerType layerType;

        public override string ToString()
        {
            string ret = "";
            ret += string.Format(
                "layerType = {0}\n" + 
                "Learning Rate = {1}\n" +
                "width = {2}\n" +
                "height = {3}\n" +
                "channels = {4}\n" +
                "filters = {5}\n" +
                "kernelSize = {6}\n" +
                "stride = {7}\n" +
                "padding = {8}\n" +
                "activation = {9}\n" +
                "hiddens = {10}\n" +
                "outputs = {11}\n" +
                "\n",
                layerType, learningRate, width, height, channels, filters,
                kernelSize, stride, padding, activation, hiddens, outputs
            );

            return ret;
        }
    }
}
