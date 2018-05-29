using System;
using System.IO;

namespace ConvNeuralNetwork
{
    enum DataSet { Training, Testing }

    class MNIST_Parser
    {
        const int MaxTrainingImageCount = 60000;
        const int MaxTestingImageCount = 10000;
        private static string path_training_images = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "MNIST", "train-images.idx3-ubyte");
        private static string path_training_labels = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "MNIST", "train-labels.idx1-ubyte");
        private static string path_test_images = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "MNIST", "t10k-images.idx3-ubyte");
        private static string path_test_labels = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "MNIST", "t10k-labels.idx1-ubyte");
        private static int imageCount;

        public static int ImageCount
        {
            get { return imageCount; }
            set
            {
                imageCount = value;
            }
        }

        public static DigitImage[] ReadFromFile(DataSet dataSet, int count = -1)
        {
            DigitImage[] digitImages=null;
            string pathImages = "";
            string pathLabels = "";
            if (count != -1)
            {
                ImageCount = count;
                digitImages = new DigitImage[ImageCount];
            }
            if (dataSet == DataSet.Training)
            {
                if (imageCount==-1)
                {
                    digitImages = new DigitImage[MaxTrainingImageCount];
                    ImageCount = MaxTrainingImageCount; 
                }

                pathImages = path_training_images;
                pathLabels = path_training_labels;
            }
            else
            {
                if (imageCount==-1)
                {
                    digitImages = new DigitImage[MaxTestingImageCount];
                    ImageCount = MaxTestingImageCount; 
                }

                pathImages = path_test_images;
                pathLabels = path_test_labels;
            }
          
            FileStream ifsLabels;
            FileStream ifsImages;
            try
            {
                //Console.WriteLine("\nBegin\n");

                ifsLabels =
                new FileStream(pathLabels,
                FileMode.Open); // test labels

                ifsImages =
                 new FileStream(pathImages,
                 FileMode.Open); // test images

                BinaryReader brLabels =
                 new BinaryReader(ifsLabels);
                BinaryReader brImages =
                 new BinaryReader(ifsImages);

                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();

                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                byte[][] pixels = new byte[28][];
                for (int i = 0; i < pixels.Length; ++i)
                    pixels[i] = new byte[28];

                // each test image
                //there are 10 000 images so you have this limit
                for (int di = 0; di < ImageCount; ++di)
                {
                    //Console.Clear();
                    for (int i = 0; i < 28; ++i)
                    {
                        for (int j = 0; j < 28; ++j)
                        {
                            byte b = brImages.ReadByte();
                            pixels[i][j] = b;
                        }
                    }

                    byte lbl = brLabels.ReadByte();

                    DigitImage dImage =
                      new DigitImage(pixels, lbl);
                    digitImages[di] = dImage;
                    //Console.WriteLine(dImage.ToString());
                    //Console.ReadLine();

                } // each image

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();

                //Console.WriteLine("\nEnd\n");
                //Console.ReadLine();
                return digitImages;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                //Console.ReadLine();

            }
            return null;
        }
    }

    class DigitImage
    {
        public byte[][] pixels;
        public byte label;

        public DigitImage(byte[][] pixels,
          byte label)
        {
            this.pixels = new byte[28][];
            for (int i = 0; i < this.pixels.Length; ++i)
                this.pixels[i] = new byte[28];

            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    this.pixels[i][j] = pixels[i][j];

            this.label = label;
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (this.pixels[i][j] == 0)
                        s += " "; // white
                    else if (this.pixels[i][j] == 255)
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            s += this.label.ToString();
            return s;
        } // ToString
    }
}

