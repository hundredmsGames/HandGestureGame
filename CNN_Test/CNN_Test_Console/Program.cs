using System;
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
            cnn.Train(null, new Matrix(new double[]{ 2.0, 3.0 }));

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
