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
        public int layers;
        public List<int> neurons;
        public List<ActivationType> fc_activations;

        public LayerType layerType;

        public override string ToString()
        {
            string ret = "";
            ret += string.Format(
                "layerType = {0}\n" +
                "width = {1}\n" +
                "height = {2}\n" +
                "channels = {3}\n" +
                "filters = {4}\n" +
                "kernelSize = {5}\n" +
                "stride = {6}\n" +
                "padding = {7}\n",
                layerType, width, height, channels, filters,
                kernelSize, stride, padding
            );

            if (neurons != null)
            {
                ret += "neurons = ";
                foreach (int i in neurons)
                    ret += i + ", ";
                ret = ret.TrimEnd(',', ' ');
                ret += "\n";
            }

            if (fc_activations != null)
            {
                ret += "activations = ";
                foreach(ActivationType act in fc_activations)
                    ret += act.ToString() + ", ";
                ret = ret.TrimEnd(',', ' ');
                ret += "\n";
            }
            else
            {
                ret += string.Format("activation = {0}\n", activation);
            }

            return ret;
        }
    }
}
