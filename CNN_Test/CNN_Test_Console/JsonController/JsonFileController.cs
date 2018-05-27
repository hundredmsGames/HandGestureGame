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

        //static public void WriteTweetsToTxtFile(TweetData[] tweetDatas,string txtFilePath)
        //{
        //    using (StreamWriter writer =
        //    new StreamWriter(txtFilePath))
        //    {
        //        for (int i = 0; i < tweetDatas.Length; i++)
        //        {
        //            string processedTweet = "";
        //            for (int j = 0; j < tweetDatas[i].words.Length; j++)
        //            {
        //                processedTweet += tweetDatas[i].words[j] + " ";
        //            }

        //            processedTweet.TrimEnd();

        //            writer.WriteLine(string.Format("raw tweet: {0}\n\n\n\n\n", tweetDatas[i].tweet));
        //            writer.WriteLine();
        //            writer.WriteLine(string.Format("processed tweet: {0}", processedTweet));
        //            writer.WriteLine();
        //            writer.WriteLine(string.Format("label: {0}", tweetDatas[i].users[0].labels));
        //            writer.WriteLine();
        //            writer.WriteLine();
        //        }
        //    }

        //}

        //static public void CombineAllJsonFiles(string path)
        //{
        //    List<TweetData> allTweets = new List<TweetData>();
        //    TweetData[] tweets = null;

        //    string[] files = Directory.GetFiles(path, "*.json");

        //    foreach (string fileName in files)
        //    {
        //        //if (fileName.EndsWith(".json") == false)
        //        //    continue;

        //        tweets = ReadDataFromJsonFile<TweetData[]>(fileName);
        //        if (fileName != "all_tweets.json")
        //            File.Delete(fileName);
        //        allTweets.AddRange(tweets);
        //    }
        //    if (allTweets.Count != 0)
        //        WriteToJsonFile(Path.Combine(path, "all_tweets.json"), allTweets.ToArray());
        //}

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
               
                //Sending into JsonConvert.DeserializeObject for getting the data as TweetData array
                dataArr = JsonConvert.DeserializeObject<T>(context);
                
                //close the file
                textReader.Close();
            }

            return (T)dataArr;
        }
    }
}
