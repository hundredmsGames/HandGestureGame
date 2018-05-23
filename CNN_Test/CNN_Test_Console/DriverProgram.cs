using System;
using System.Diagnostics;
using ConvNeuralNetwork;
using MatrixLib;

namespace CNN_Test_Console
{
    class DriverProgram
    {
        static int cursorTop = 0;

        static void Main(string[] args)
        {
            //CNN_Test();
            //CNN_OverfittingTest();

            CNN cnn = new CNN();

            Matrix[] input = new Matrix[1];
            input[0] = new Matrix(8, 8);
            input[0].Randomize();

            Matrix target = new Matrix(3, 1);
            target.Randomize();

            cnn.Train(input, target);

            Console.WriteLine("DONE");
            Console.ReadLine();
        }

        public static void CNN_Test()
        {
            DigitImage[] digitImages = MNIST_Parser.ReadFromFile(DataSet.Training);
            int training_count = digitImages.Length;

            CNN cnn = new CNN();
            Matrix[] input = new Matrix[1];
            input[0]= new Matrix(28, 28);
            Matrix[] targets = new Matrix[10];

            for (int i = 0; i < 10; i++)
            {
                targets[i] = new Matrix(10, 1);
                for (int j = 0; j < 10; j++)
                {
                    targets[i][j, 0] = (i == j) ? 1.0f : 0.0f;
                }
            }

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            Console.WriteLine("System is getting trained...");
            cursorTop = Console.CursorTop;

            for (int i = 0; i < training_count; i++)
            {
                for (int j = 0; j < 28; j++)
                    for (int k = 0; k < 28; k++)
                        input[0][j, k] = digitImages[i].pixels[j][k];

                input[0].Normalize(0.0f, 255.0f, 0.0f, 1.0f);
                cnn.Train(input, targets[digitImages[i].label]);

                //Y = (X-A)/(B-A) * (D-C) + C
                int val = (int)((i - 0) / (double)(training_count - 1 - 0) * (100 - 0) + 0);
                ProgressBar(val, i, training_count, stopwatch.ElapsedMilliseconds / 1000.0);
            }

            digitImages = MNIST_Parser.ReadFromFile(DataSet.Testing);
            int testing_count = digitImages.Length;
            int correct_count = 0;

            Console.WriteLine("\nSystem has been trained.");
            Console.WriteLine("System is getting tested. You will see the results when it is done...\n");
            cursorTop = Console.CursorTop;

            for (int i = 0; i < testing_count; i++)
            {
                for (int j = 0; j < 28; j++)
                    for (int k = 0; k < 28; k++)
                        input[0][j, k] = digitImages[i].pixels[j][k];

                input[0].Normalize(0.0f, 255.0f, 0.0f, 1.0f);
                Matrix ans = cnn.Predict(input);

                if (ans.GetMaxRowIndex() == digitImages[i].label)
                    correct_count++;


                int val = (int)((i - 0) / (double)(testing_count - 1 - 0) * (100 - 0) + 0);
                ProgressBar(val, i, testing_count, stopwatch.ElapsedMilliseconds / 1000.0);
            }

            Console.WriteLine("\nTime :" + (stopwatch.ElapsedMilliseconds / 1000.0).ToString("F4"));
            Console.WriteLine("\nAccuracy: %{0:F2}\n", (correct_count * 1f / testing_count) * 100.0);
            Console.WriteLine("Correct/All: {0}/{1}", correct_count, testing_count);

            Console.ReadLine();
        }

        public static void CNN_OverfittingTest()
        {
            CNN cnn = new CNN();

            DigitImage[] digitImages = MNIST_Parser.ReadFromFile(DataSet.Testing, 100);

            int test_image_idx = 0;
            Matrix[] input = new Matrix[1];
            input[0] = new Matrix(digitImages[test_image_idx].pixels);
            Matrix target = new Matrix(10, 1);
            target[(int) digitImages[test_image_idx].label, 0] = 1f;

            int iteration_count = 1000;
            for(int i = 0; i < iteration_count; i++)
            {
                cnn.Train(input, target);
               
                int val = (int)((i - 0) / (double)(iteration_count - 1 - 0) * (100 - 0) + 0);
                ProgressBar(val,0,0);
            }

            Matrix output = cnn.Predict(input);

            Console.WriteLine("Output");
            Console.WriteLine(output.ToString());

            Console.WriteLine(digitImages[test_image_idx].ToString());
        }

        static void ProgressBar(int currentValue,int currentCount,int maxCount,double timePassed=0)
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
            Console.SetCursorPosition(25, cursorTop);
            Console.WriteLine(currentCount+1 + " / " + maxCount);
            Console.SetCursorPosition(40, cursorTop);
            Console.WriteLine("Time Passed:" + timePassed.ToString("F2"));
            Console.CursorVisible = true;
        }
    }
}

