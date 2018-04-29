using System;
using System.IO;
using System.Windows.Forms;
using ConvNeuralNetwork;
using FullyConnectedNN;
using MatrixLib;

namespace CNN_Test_Console
{

    class Program
    {

        static void Main(string[] args)
        {



            CNN cnn = new CNN();

            DigitImage[] digitImages = MNIST_Parser.ReadFromFile();


            for (int i = 0; i < digitImages.Length; i++)
            {
                double[] target = new double[10];

                target[(int)(digitImages[i].label)] = 1;
                Matrix inMatrix = new Matrix(digitImages[i].pixels);
                cnn.Train(inMatrix, new Matrix(target));
            }


            Console.WriteLine("End");

            //cnn.Train(input, new Matrix(output));

            //Matrix.Normalize(new Matrix(/*Buraya verimiz gelecek ve bu metod geri normalized matrix döndürecek*/),/*other vars*/);
            //MNIST_Parser.ReadFromFile();
            //FCNN_Test();

            Console.ReadLine();
        }

        public static void FCNN_Test()
        {
            FCNN fcnn = new FCNN(3, 10, 3, 0.2, FCNN.Sigmoid, FCNN.DerSigmoid);

            Matrix input = new Matrix(new double[] { 1, 2, 3 });
            Matrix output = new Matrix(new double[] { 2, 3, 4 });

            Matrix o = fcnn.FeedForward(input);
            fcnn.Train(input, output);

            Console.WriteLine(o.ToString());

            o = fcnn.FeedForward(input);

            Console.WriteLine(o.ToString());
        }
    }

}

