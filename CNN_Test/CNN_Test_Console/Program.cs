using System;
using System.Diagnostics;
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
            Stopwatch stopwatch = new Stopwatch();
            
            CNN cnn = new CNN();

            DigitImage[] digitImages = MNIST_Parser.ReadFromFile();

            double[][] outputs = new double[10][];
            for(int i = 0; i < 10; i++)
            {
                outputs[i] = new double[10];
                for(int j = 0; j < 10; j++)
                {
                    outputs[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }

            stopwatch.Start();

            double[] target;

            int training_count = (int) (digitImages.Length * 0.7);

            for (int i = 0; i < training_count; i++)
            {
                target = outputs[(int)(digitImages[i].label)];

                Matrix inMatrix = new Matrix(digitImages[i].pixels);
                cnn.Train(inMatrix, new Matrix(target));
            }

            int correct_count = 0;

            for (int i = training_count + 1; i < digitImages.Length; i++)
            {
                Matrix ans = cnn.Predict(new Matrix(digitImages[i].pixels));
                if(ans[digitImages[i].label, 0] > 0.7)
                {
                    correct_count++;
                }
            }

            Console.WriteLine("\nEnd");
            Console.WriteLine((stopwatch.ElapsedMilliseconds/1000.0).ToString("F4"));
            Console.WriteLine("\n%{0}\n", (correct_count * 1f / (digitImages.Length - training_count)) * 100);
            Console.WriteLine("{0}/{1}", correct_count, digitImages.Length - training_count);


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

