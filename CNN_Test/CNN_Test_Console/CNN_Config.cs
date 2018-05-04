using System;
using System.IO;

namespace ConvNeuralNetwork
{
    partial class CNN
    {
        string path = Path.Combine("..", "..", "config_files", "config.cfg");

        #region Deserialization

        public void Deserialize()
        {
            StreamReader streamReader = new StreamReader(path);

            while (streamReader.EndOfStream == false)
            {
                string line = streamReader.ReadLine();

                string[] parameter = line.Split('=');
                switch (parameter[0])
                {
                    case "learning_rate":
                        learning_rate = double.Parse(parameter[1].Replace('.', ','));
                        break;

                    case "fcnn_hiddens":
                        fcnn_hidden_neurons = int.Parse(parameter[1]);
                        break;
                    case "fcnn_outputs":
                        fcnn_output_neurons = int.Parse(parameter[1]);
                        break;

                    case "size":
                        l1_kernel_size= int.Parse(parameter[1]);
                        break;
                    case "stride":
                        l1_stride= int.Parse(parameter[1]);
                        break;

                    case "mp_size":
                        l2_kernel_size= int.Parse(parameter[1]);
                        break;
                    case "mp_stride":
                        l2_stride= int.Parse(parameter[1]);
                        break;
                    case "activation":
                        if(parameter[1] == "relu")
                        {
                            activation = ReLu;
                            derOfActivation = DerOfReLu;
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
