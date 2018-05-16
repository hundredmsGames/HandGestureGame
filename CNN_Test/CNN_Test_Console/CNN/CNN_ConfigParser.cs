using System;
using System.Collections.Generic;
using System.IO;


namespace ConvNeuralNetwork
{
    partial class CNN
    {
        #region Config File Paths

        string main_cfg_path = Path.Combine("..", "..", "config_files", "config.cfg");

        #endregion

        #region Deserialization

        public Description[] DeserializeConfig(bool test = false)
        {
            string path = main_cfg_path;
            StreamReader streamReader = new StreamReader(path);

            List<Description> descriptions = new List<Description>();
            Description currDesc = new Description();

            while (streamReader.EndOfStream == false)
            {
                string line = streamReader.ReadLine().Trim();

                // Comment line, continue
                if (line.StartsWith("#") == true || string.IsNullOrEmpty(line) == true)
                    continue;

                switch (line)
                {
                    case "[net]":
                        if (descriptions.Count == 0)
                            currDesc.layerType = LayerType.INPUT;

                        // otherwise throw exception

                        continue;

                    case "[convolutional]":
                        descriptions.Add(currDesc);

                        currDesc = new Description();
                        currDesc.layerType = LayerType.CONVOLUTIONAL;
                        continue;

                    case "[maxpooling]":
                        descriptions.Add(currDesc);

                        currDesc = new Description();
                        currDesc.layerType = LayerType.MAXPOOLING;
                        continue;

                    case "[fclayer]":
                        descriptions.Add(currDesc);

                        currDesc = new Description();
                        currDesc.layerType = LayerType.FULLY_CONNECTED;
                        continue;
                }
                
                string[] temp = line.Split('=');
                string param = temp[0].Trim();
                string value = temp[1].Trim();

                switch (param)
                {
                    case "width":
                        currDesc.width = int.Parse(value);
                        continue;

                    case "height":
                        currDesc.height = int.Parse(value);
                        continue;

                    case "channels":
                        currDesc.channels = int.Parse(value);
                        continue;

                    case "learning_rate":
                        currDesc.learningRate = float.Parse(value.Replace('.', ','));
                        continue;

                    case "filters":
                        currDesc.filters = int.Parse(value);
                        continue;

                    case "size":
                        currDesc.kernelSize = int.Parse(value);
                        continue;

                    case "stride":
                        currDesc.stride = int.Parse(value);
                        continue;

                    case "activation":
                        currDesc.activation = (ActivationType) Enum.Parse(typeof(ActivationType), value, true);
                        continue;

                    case "hiddens":
                        currDesc.hiddens = int.Parse(value);
                        continue;

                    case "outputs":
                        currDesc.outputs = int.Parse(value);
                        continue;

                    default:
                        // Config parser exception
                        break;
                }
            }

            // Add last description to list
            descriptions.Add(currDesc);

            return descriptions.ToArray();
        }

        #endregion
    }
}
