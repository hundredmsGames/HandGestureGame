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

    struct LayerDescription
    {
        // input desc
        public int width;
        public int height;
        public int channels;

        // conv & pool desc
        public int kernel_size;
        public int stride;
        public int padding;

        // fc layer
        public int hiddenNeuronsCount;
        public int inputNeuronsCount;
        public int outputNeuronsCount;
        public float learningRate;

        public Func<double, double> activationFunc;
        public Func<double, double> derofActivationFunc;

        public LayerType layerType;
    }
}
