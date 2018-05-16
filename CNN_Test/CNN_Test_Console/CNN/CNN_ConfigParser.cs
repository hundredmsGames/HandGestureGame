using System;
using System.IO;

namespace ConvNeuralNetwork
{
    partial class CNN
    {
        #region Config File Paths

        string main_cfg_path = Path.Combine("..", "..", "config_files", "more_layer.cfg");
        string test_cfg_path = Path.Combine("..", "..", "config_files", "test.cfg");

        #endregion

        #region Deserialization

        public LayerDescription[] DeserializeConfig(bool test = false)
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
                        learningRate = float.Parse(parameter[1].Trim().Replace('.', ','));
                        break;

                    default:
                        // Config parser exception
                        break;
                }
            }

            return null;
        }

        #endregion
    }
}
