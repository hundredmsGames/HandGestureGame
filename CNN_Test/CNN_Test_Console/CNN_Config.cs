using System;
using System.IO;

namespace ConvNeuralNetwork
{
    partial class CNN
    {
        string main_cfg_path = Path.Combine("..", "..", "config_files", "more_layer.cfg");
        string test_cfg_path = Path.Combine("..", "..", "config_files", "test.cfg");

        #region Deserialization

        public void Deserialize(bool test = false)
        {
            string path = test ? test_cfg_path : main_cfg_path;
            StreamReader streamReader = new StreamReader(path);

            while (streamReader.EndOfStream == false)
            {
                string line = streamReader.ReadLine();

                string[] parameter = line.Split('=');
                switch (parameter[0].Trim())
                {
                    case "learning_rate":
                        learning_rate = double.Parse(parameter[1].Trim().Replace('.', ','));
                        break;

                    case "fcnn_hiddens":
                        fcnn_hidden_neurons = int.Parse(parameter[1].Trim());
                        break;
                    case "fcnn_outputs":
                        fcnn_output_neurons = int.Parse(parameter[1].Trim());
                        break;

                    case "conv1_size":
                        l1_kernel_size= int.Parse(parameter[1].Trim());
                        break;
                    case "conv1_stride":
                        l1_stride= int.Parse(parameter[1].Trim());
                        break;

                    case "conv2_size":
                        l2_kernel_size = int.Parse(parameter[1].Trim());
                        break;
                    case "conv2_stride":
                        l2_stride = int.Parse(parameter[1].Trim());
                        break;

                    case "mp_size":
                        l3_kernel_size= int.Parse(parameter[1].Trim());
                        break;
                    case "mp_stride":
                        l3_stride= int.Parse(parameter[1].Trim());
                        break;
                    case "activation":
                        if(parameter[1].Trim() == "relu")
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
