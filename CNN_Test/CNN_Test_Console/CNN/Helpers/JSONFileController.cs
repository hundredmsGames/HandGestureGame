using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace ConvNeuralNetwork
{
    static class JsonFileController
    {
        static public void WriteToJsonFile(string path, object obj)
        {
            using (TextWriter textWriter = new StreamWriter(path))
            {
                //create a new json serializer object for serialazation
                JsonSerializer jsonSerializer = new JsonSerializer();

                //serialize data
                string jsonText = JsonConvert.SerializeObject(obj,new JsonSerializerSettings() { ReferenceLoopHandling = ReferenceLoopHandling.Ignore });
                textWriter.Write(jsonText);

                textWriter.Close();
            }
        }

        /// <summary>
        /// reads data from a json file we can specify what type of data we will read
        /// </summary>
        /// <param name="path">file path</param>
        static public T ReadDataFromJsonFile<T>(string path)
        {
            object dataArr;

            //we wont need text reader after this scope so I am being sure GC has collected this ref.
            using (TextReader textReader = new StreamReader(path, Encoding.UTF8))
            {
                //Reading the json File
                string context = textReader.ReadToEnd();
                JsonSerializerSettings jsonSerializerSettings= new JsonSerializerSettings();
               
                dataArr = JsonConvert.DeserializeObject<T>(context);
                
                //close the file
                textReader.Close();
            }

            return (T)dataArr;
        }
    }
}
