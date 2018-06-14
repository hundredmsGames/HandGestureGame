using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ConvNeuralNetwork
{
    partial class CNN
    {
        #region Config File Paths

        string main_cfg_path = Path.Combine("..", "..", "CNN", "Configs", "config.cfg");
        string saved_network_path = Path.Combine("..", "..", "CNN", "Configs");

        #endregion

        #region Deserialization from config file

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
                        else
                            // otherwise throw exception
                            throw new WrongLayerException("There can be only one input layer!!!!");
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

                    case "[fc_network]":
                        // TODO: read next line, get layer count.
                        descriptions.Add(currDesc);

                        currDesc = new Description();
                        currDesc.layerType = LayerType.FULLY_CONNECTED;
                        continue;

                    case "[fc_layer]":
                        //just continue
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
                        learningRate = double.Parse(value.Replace('.', ','));
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
                        if (currDesc.layerType != LayerType.FULLY_CONNECTED)
                            currDesc.activation = (ActivationType)Enum.Parse(typeof(ActivationType), value, true);
                        else
                        {
                            if (currDesc.fc_activations == null)
                                currDesc.fc_activations = new List<ActivationType>();
                            
                            currDesc.fc_activations.Add((ActivationType)Enum.Parse(typeof(ActivationType), value, true));
                        }
                        continue;

                    case "neurons":
                        if (currDesc.neurons == null)
                            currDesc.neurons = new List<int>();

                        currDesc.neurons.Add(int.Parse(value));
                        continue;

                    case "layers":
                        currDesc.layers = int.Parse(value);
                        continue;

                    default:
                        // Config parser exception
                        throw new ConfigParserException("token is not recognizeable...TOKEN : " + param);

                }
            }

            // Add last description to list
            descriptions.Add(currDesc);

            return descriptions.ToArray();
        }

        #endregion

        #region Save-Load Network (JSON)

        public void SaveData(string fileName)
        {
            CNN_Data cNN_Data = new CNN_Data();
            for (int i = 0; i < Layers.Length; i++)
            {
                switch (Layers[i].LayerType)
                {
                    case LayerType.INPUT:
                        break;
                    case LayerType.CONVOLUTIONAL:
                        cNN_Data.Kernels.Add((layers[i] as ConvLayer).Kernels);
                        cNN_Data.Ker_Biases.Add((layers[i] as ConvLayer).Biases);
                        break;
                    case LayerType.MAXPOOLING:
                        break;
                    case LayerType.FULLY_CONNECTED:
                        for (int j = 0; j < (layers[i] as FC_Network).Layers.Length; j++)
                        {
                            cNN_Data.Weights.Add((layers[i] as FC_Network).Layers[j].Weights);
                            cNN_Data.Biases.Add((layers[i] as FC_Network).Layers[j].Biases); 
                        }
                        break;
                    default:
                        break;
                }
            }

            cNN_Data.Descriptions = descriptions.ToList();

            int fileCount = Directory.GetFiles(saved_network_path, "*.json").Length;
            if (string.IsNullOrEmpty(fileName))
                fileName = string.Format("network{0}.json", fileCount);

            string fullPath = Path.Combine(saved_network_path, fileName);

            JsonFileController.WriteToJsonFile(fullPath, cNN_Data);
        }

        public void LoadData(string fileName)
        {
            int convLayCount = 0;

            CNN_Data cNN_Data = JsonFileController.ReadDataFromJsonFile<CNN_Data>(Path.Combine(saved_network_path,fileName));
            descriptions = cNN_Data.Descriptions.ToArray();
            layers = new Layer[descriptions.Length];
            for (int i = 0; i < descriptions.Length; i++)
            {
                NewLayer(descriptions[i]);
                switch (descriptions[i].layerType)
                {
                    case LayerType.INPUT:
                        break;
                    case LayerType.CONVOLUTIONAL:
                        (layers[i] as ConvLayer).Kernels = cNN_Data.Kernels[convLayCount];
                        (layers[i] as ConvLayer).Biases = cNN_Data.Ker_Biases[convLayCount];
                        convLayCount++;
                        break;
                    case LayerType.MAXPOOLING:
                        break;
                    case LayerType.FULLY_CONNECTED:

                        for (int j = 0; j < (layers[i] as FC_Network).Layers.Length; j++)
                        {
                            (layers[i] as FC_Network).Layers[j].Weights = cNN_Data.Weights[j];
                            (layers[i] as FC_Network).Layers[j].Biases  = cNN_Data.Biases[j];
                        }
                        break;
                    default:
                        throw new UndefinedLayerException("This is not a recognizeable layer!!!");
                }
            }
        }

        #endregion
    }
}
