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
        public int neurons;

        public LayerType layerType;

        public override string ToString()
        {
            string ret = "";
            ret += string.Format(
                "layerType = {0}\n" + 
                "width = {2}\n" +
                "height = {3}\n" +
                "channels = {4}\n" +
                "filters = {5}\n" +
                "kernelSize = {6}\n" +
                "stride = {7}\n" +
                "padding = {8}\n" +
                "activation = {9}\n" +
                "neurons = {10}\n" +
                "\n",
                layerType, width, height, channels, filters,
                kernelSize, stride, padding, activation, neurons
            );

            return ret;
        }
    }
}
