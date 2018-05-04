using System;
using System.Diagnostics;
using ConvNeuralNetwork;
using FullyConnectedNN;
using MatrixLib;

namespace CNN_Test_Console
{

    class DriverProgram
    {
        static int cursorTop = 0;

        static void Main(string[] args)
        {
            //CNN_OverfittingTest();
            CNN_Test();
    
            Console.ReadLine();
        }

        static void CNN_OverfittingTest()
        {
            DigitImage[] digitImages = MNIST_Parser.ReadFromFile(50);

            CNN cnn = new CNN();

            int test_image_idx = 3;
            Matrix input = new Matrix(digitImages[test_image_idx].pixels);
            Matrix target = new Matrix(10, 1);
            target[(int) digitImages[test_image_idx].label, 0] = 1.0;

            int iteration_count = 1000;
            for(int i = 0; i < iteration_count; i++)
            {
                cnn.Train(input, target);

                int val = (int)((i - 0) / (double)(iteration_count - 1 - 0) * (100 - 0) + 0);
                ProgressBar(val);
            }

            Matrix output = cnn.Predict(input);

            Console.WriteLine("Output");
            Console.WriteLine(output.ToString());

            Console.WriteLine(digitImages[test_image_idx].ToString());
        }

        static void CNN_Test()
        {
            DigitImage[] digitImages = MNIST_Parser.ReadFromFile(50000);

            CNN cnn = new CNN();
            Matrix input = new Matrix(28, 28);
            Matrix[] targets = new Matrix[10];

            for (int i = 0; i < 10; i++)
            {
                targets[i] = new Matrix(10, 1);
                for (int j = 0; j < 10; j++)
                {
                    targets[i][j, 0] = (i == j) ? 1.0 : 0.0;
                }
            }

            int training_count = (int)(digitImages.Length * 0.8);
            cursorTop = Console.CursorTop;

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            for (int i = 0; i < training_count; i++)
            {
                for (int j = 0; j < 28; j++)
                    for (int k = 0; k < 28; k++)
                        input[j, k] = digitImages[i].pixels[j][k];


                input.Normalize(0.0, 255.0, 0.0, 1.0);
                cnn.Train(input, targets[digitImages[i].label]);

                //Y = (X-A)/(B-A) * (D-C) + C
                int val = (int)((i - 0) / (double)(training_count - 1 - 0) * (100 - 0) + 0);
                ProgressBar(val);
            }

            int correct_count = 0;
            int testing_count = digitImages.Length - training_count;
            cursorTop = Console.CursorTop;

            for (int i = training_count; i < digitImages.Length; i++)
            {
                for (int j = 0; j < 28; j++)
                    for (int k = 0; k < 28; k++)
                        input[j, k] = digitImages[i].pixels[j][k];

                input.Normalize(0.0, 255.0, 0.0, 1.0);
                Matrix ans = cnn.Predict(input);

                if (ans.GetMaxRowIndex() == digitImages[i].label)
                    correct_count++;
                

                int val = (int)((i - training_count - 0) / (double)(testing_count - 1 - 0) * (100 - 0) + 0);
                ProgressBar(val);
            }

            Console.WriteLine("\nTime :" + (stopwatch.ElapsedMilliseconds / 1000.0).ToString("F4"));
            Console.WriteLine("\nAccuracy: %{0:F2}\n", (correct_count * 1f / testing_count) * 100.0);
            Console.WriteLine("Correct/All: {0}/{1}", correct_count, testing_count);
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
      
        static void ProgressBar(int currentValue)
        {
            Console.CursorVisible = false;

            int pos = currentValue / 10;
            if (currentValue == 0)
            {
                Console.SetCursorPosition(0, cursorTop);
                Console.Write("[");
                Console.SetCursorPosition(pos + 12, cursorTop);
                Console.Write("]");
            }

            Console.SetCursorPosition(pos+1, cursorTop);
            Console.Write("#");
            Console.SetCursorPosition(14, cursorTop);
            Console.WriteLine(currentValue + "%");

            Console.CursorVisible = true;
        }
    }

}

