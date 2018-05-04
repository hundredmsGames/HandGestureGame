using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNeuralNetwork
{
    partial class CNN
    {
        /*
            [net]
            width=28
            height=28

            learning_rate=0.04
            fcnn_hiddens=100
            fcnn_outputs=10


            [convolutinal]
            size=5
            stride=2
            mp_size=2
            mp_stride=2
            activation=relu
             
             */
        #region Deserialization

        public void Deserialize(string path)
        {
            StreamReader streamReader = new StreamReader("config.cfg");
            while (streamReader.EndOfStream == false)
            {
                string line = streamReader.ReadLine();
                switch (line)
                {
                    case "learning_rate":
                        learning_rate =double.Parse(line.Split('=')[1]);
                        break;

                    case "fcnn_hiddens":
                        fcnn_hidden_neurons = int.Parse(line.Split('=')[1]);
                        break;
                    case "fcnn_outputs":
                        fcnn_output_neurons = int.Parse(line.Split('=')[1]);
                        break;

                    case "size":
                        l1_kernel_size= int.Parse(line.Split('=')[1]);
                        break;
                    case "stride":
                        l1_stride= int.Parse(line.Split('=')[1]);
                        break;

                    case "mp_size":
                        l2_kernel_size= int.Parse(line.Split('=')[1]);
                        break;
                    case "mp_stride":
                        l2_stride= int.Parse(line.Split('=')[1]);
                        break;
                    case "activation":
                        if(line.Split('=')[1]== "ReLu")
                        {
                            activation = ReLu;
                            derOfActivation = DerOfReLu;
                        }
                        else if (line.Split('=')[1] == "Another")
                        {
                            //activation=another;
                        }
                        break;
                    default:

                        break;
                }


            }

        }

        #endregion
    }
}
