using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNeuralNetwork
{
    class CSV_Helper
    {
        private static string trainPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "CNN", "KAGGLE", "train.csv");
        private static string testPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "CNN", "KAGGLE", "test.csv");

        public static DigitImage[] Read(DataSet dataset)
        {
            FileStream fileStream;
            if (dataset == DataSet.Training)
                fileStream = new FileStream(trainPath, FileMode.Open, FileAccess.Read);
            else
                fileStream = new FileStream(testPath, FileMode.Open, FileAccess.Read);

            List<DigitImage> digitImageList = new List<DigitImage>();
            using (var streamReader = new StreamReader(fileStream, Encoding.UTF8))
            {
                string line;
                streamReader.ReadLine();
             
                while ((line = streamReader.ReadLine()) != null)
                {
                    string[] splitted = line.Split(',');

                    byte[][] pixels = new byte[28][];
                    for (int i = 0; i < pixels.Length; ++i)
                        pixels[i] = new byte[28];

                    for(int i = 0; i < splitted.Length - 1; ++i)
                    {
                        pixels[i / 28][i % 28] = (byte) int.Parse(splitted[i + 1]);
                    }

                    byte label = (dataset == DataSet.Training ? byte.Parse(splitted[0]) : (byte)10);
                    DigitImage d = new DigitImage(pixels, label);
                    digitImageList.Add(d);
                }
            }

            return digitImageList.ToArray();
        }

        public static void Write(string content)
        {
            File.AppendAllText(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "submission.csv"),
                        content + Environment.NewLine);
        }
    }
}
