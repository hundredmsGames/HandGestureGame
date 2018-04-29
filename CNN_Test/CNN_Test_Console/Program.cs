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
            Random r = new Random(12312324);

            Matrix input = new Matrix(28, 28);
            for (int i = 0; i < input.rows; i++)
            {
                for (int j = 0; j < input.cols; j++)
                {
                    input[i, j] = r.NextDouble() * 2f - 1f;
                }
            }


            CNN cnn = new CNN();   

            double[] output = new double[10];
            for (int i = 0; i < 10; i++)
            {
                output[i] = i;
            }

            cnn.Train(input, new Matrix(output));

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

