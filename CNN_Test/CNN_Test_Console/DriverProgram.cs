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
            CNN_Test();
            //CNN_OverfittingTest();

            Console.ReadLine();
        }

        public static void CNN_Test()
        {
            int trCount = 100, tsCount = 100;
            float error = 0f;
            float timeLimit = 0;
            int iterationCount = 10;
            bool predictionIsOn = true;
            char dialogResult;

            DigitImage[] digitImages = MNIST_Parser.ReadFromFile(DataSet.Training, trCount);
            int training_count = digitImages.Length;
            CNN cnn;

            Console.WriteLine("Would you like to load data from file?(Yy/Nn)");
            dialogResult = Console.ReadLine().Trim().ToLower()[0];
            if (dialogResult == 'y')
                cnn = new CNN(false);
            else
                cnn = new CNN();

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
            if (dialogResult != 'y')
            {
                Console.WriteLine("System is getting trained...");
                cursorTop = Console.CursorTop;
                for (int x = 0; x < iterationCount; x++)
                {
                    for (int i = 0; i < training_count; i++)
                    {
                        for (int j = 0; j < 28; j++)
                            for (int k = 0; k < 28; k++)
                                input[0][j, k] = digitImages[i].pixels[j][k];

                        input[0].Normalize(0f, 255f, 0f, 1f);
                        cnn.Train(input, targets[digitImages[i].label]);

                        if (stopwatch.ElapsedMilliseconds > timeLimit)
                        {
                            // every 0.5 sec update error
                            timeLimit += 500;
                            error = cnn.GetError();
                        }

                        int val = Map(0, training_count * iterationCount, 0, 100, training_count * x + i);
                        ProgressBar(val, training_count * x + i, training_count * iterationCount, error, stopwatch.ElapsedMilliseconds / 1000.0);
                    }
                }

               

                Console.WriteLine("\nSystem has been trained.");
            }

            digitImages = MNIST_Parser.ReadFromFile(DataSet.Testing, tsCount);
            int testing_count = digitImages.Length;
            int correct_count = 0;
            Console.WriteLine("System is getting tested. You will see the results when it is done...\n");
            cursorTop = Console.CursorTop;
            timeLimit = stopwatch.ElapsedMilliseconds;

            for (int i = 0; i < testing_count; i++)
            {
                for (int j = 0; j < 28; j++)
                    for (int k = 0; k < 28; k++)
                        input[0][j, k] = digitImages[i].pixels[j][k];

                input[0].Normalize(0f, 255f, 0f, 1f);

                Matrix ans=null;
                if (predictionIsOn)
                    ans = cnn.Predict(input);
                else
                {
                   
                    cnn.Train(input, targets[digitImages[i].label]);
                    ans = cnn.Layers[cnn.Layers.Length - 1].Output[0];
                }

                if (ans.GetMaxRowIndex() == digitImages[i].label)
                    correct_count++;


                if (stopwatch.ElapsedMilliseconds > timeLimit)
                {
                    // every 0.5 sec update error
                    timeLimit += 500;
                    error = cnn.GetError();
                }

                int val = Map(0, testing_count, 0, 100, i);
                ProgressBar(val, i, testing_count, error, stopwatch.ElapsedMilliseconds / 1000.0);
            }

            Console.WriteLine("\nTime :" + (stopwatch.ElapsedMilliseconds / 1000.0).ToString("F4"));
            Console.WriteLine("\nAccuracy: %{0:F2}\n", (correct_count * 1f / testing_count) * 100.0);
            Console.WriteLine("Correct/All: {0}/{1}", correct_count, testing_count);

            Console.WriteLine(cnn.Layers[cnn.Layers.Length - 1].Output[0]);

            //if we are not training the system we dont have to save again
            if (dialogResult != 'y')
            {
                Console.WriteLine("Would you like to save this network?(yY/nN)");
                dialogResult = Console.ReadLine().ToLower()[0];
                if (dialogResult == 'y')
                {
                    cnn.SaveData();
                }
                Console.WriteLine("Data Saved...");
            }
            Console.WriteLine("Press enter to exit.");
        }
        
        public static void CNN_OverfittingTest()
        {
            CNN cnn = new CNN();

            DigitImage[] digitImages = MNIST_Parser.ReadFromFile(DataSet.Testing, 100);

            int test_image_idx = 5;
            Matrix[] input = new Matrix[1];
            input[0] = new Matrix(digitImages[test_image_idx].pixels);
            Matrix target = new Matrix(10, 1);
            target[(int) digitImages[test_image_idx].label, 0] = 1f;

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            int iteration_count = 10000;
            for(int i = 0; i < iteration_count; i++)
            {
                cnn.Train(input, target);
                float error = cnn.GetError();
               
                int val = (int)((i - 0) / (double)(iteration_count - 1 - 0) * (100 - 0) + 0);
                ProgressBar(val, i, iteration_count, error, stopwatch.ElapsedMilliseconds / 1000.0);
            }

            Matrix output = cnn.Predict(input);

            Console.WriteLine(output.ToString());
            Console.WriteLine(digitImages[test_image_idx].ToString());
        }

        public static int Map(int oldMin, int oldMax, int newMin, int newMax, int current)
        {
            //Y = (X-A)/(B-A) * (D-C) + C
            return (int)((current - oldMin) / (double)(oldMax - 1 - oldMin) * (newMax - newMin) + newMin);
        }

        static void ProgressBar(int currentValue, int currentCount, int maxCount, float error, double timePassed = 0)
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
            Console.SetCursorPosition(45, cursorTop);
            Console.WriteLine("Time Passed: " + timePassed.ToString("F1"));
            Console.SetCursorPosition(70, cursorTop);
            Console.WriteLine("Error: {0:F6}", error);

            Console.CursorVisible = true;
        }
    }
}

