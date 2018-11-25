using System;
using System.Diagnostics;
using ConvNeuralNetwork;
using MatrixLib;
using System.Linq;
using System.IO;

namespace CNN_Test_Console
{
    class DriverProgram
    {
        static int cursorTopTraining = -1, cursorTopTesting = 0;

        static void Main(string[] args)
        {
            //Kaggle_Test();
            CNN_Training();
            //CNN_OverfittingTest();

            Console.ReadLine();
        }

        public static void Kaggle_Test()
        {
            //DigitImage[] training = CSV_Helper.Read(DataSet.Training);
            DigitImage[] testing = CSV_Helper.Read(DataSet.Testing);

            int dialogResult;
            CNN cnn = CreateCNN(out dialogResult);
            DigitImage[] digitImagesDatas = testing;

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            int testing_count = digitImagesDatas.Length;
            int correct_count = 0;
            double timeLimit = 0, error = 0;
            Matrix[] input, targets;

            Console.WriteLine("System is getting tested. You will see the results when it is done...\n");
            if (cursorTopTesting == 0)
                cursorTopTesting = Console.CursorTop;

            InitializeInputAndTarget(out input, out targets);
            timeLimit = stopwatch.ElapsedMilliseconds;

            string submissionCSV = "ImageId,Label" + Environment.NewLine;
            for (int i = 0; i < testing_count; i++)
            {
                for (int j = 0; j < 28; j++)
                    for (int k = 0; k < 28; k++)
                        input[0][j, k] = digitImagesDatas[i].pixels[j][k];

                input[0].Normalize(0f, 255f, 0f, 1f);

                Matrix ans = cnn.Predict(input);

                submissionCSV += (i + 1).ToString() + "," + ans.GetMaxRowIndex() + Environment.NewLine;

                if (stopwatch.ElapsedMilliseconds > timeLimit)
                {
                    // every 0.5 sec update error
                    timeLimit += 500;
                    error = cnn.GetError();
                }

                int val = Map(0, testing_count, 0, 100, i);
                ProgressBar(val, i, testing_count, error, stopwatch.ElapsedMilliseconds / 1000.0, cursorTopTesting);
            }

            CSV_Helper.Write(submissionCSV);
        }

        public static void CNN_Training()
        {
            int trCount = 60000, tsCount = 10000;
            double error = 0f, timeLimit = 0f;

            int totalEpoch = 10;
            bool predictionIsOn = true;
            Random random = new Random();

            DigitImage[] trainigDigitImagesDatas = MNIST_Parser.ReadFromFile(DataSet.Training, trCount);
            DigitImage[] testingDigitImagesDatas = MNIST_Parser.ReadFromFile(DataSet.Testing, tsCount);
            int training_count = trainigDigitImagesDatas.Length;
            int dialogResult;

            CNN cnn = CreateCNN(out dialogResult);
            Matrix[] input, targets;
            InitializeInputAndTarget(out input, out targets);

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            if (dialogResult == 0)
            {
                Console.WriteLine("System is getting trained...");
                //if we never assigned this assign only once
                if (cursorTopTraining == -1)
                    cursorTopTraining = Console.CursorTop;

                for (int epoch = 0; epoch < totalEpoch; epoch++)
                {
                    double lossSum = 0.0;
                    DigitImage[] digitImages = trainigDigitImagesDatas.OrderBy(image => random.Next(training_count)).ToArray();
                    for (int i = 0; i < training_count; i++)
                    {
                        for (int j = 0; j < 28; j++)
                            for (int k = 0; k < 28; k++)
                                input[0][j, k] = digitImages[i].pixels[j][k];

                        input[0].Normalize(0f, 255f, 0f, 1f);
                        cnn.Train(input, targets[digitImages[i].label]);

                        //if (stopwatch.ElapsedMilliseconds > timeLimit)
                       // {
                            // every 0.5 sec update error
                            //timeLimit += 500;
                        error = cnn.GetError();
                        lossSum += error;

                        //}

                        int val = Map(0, training_count * totalEpoch, 0, 100, training_count * epoch + i);
                        ProgressBar(val, training_count * epoch + i, training_count * totalEpoch, lossSum / (i + 1), stopwatch.ElapsedMilliseconds / 1000.0, cursorTopTraining);
                    }
                    CNN_Testing(testingDigitImagesDatas, predictionIsOn, cnn, epoch + 1);
                }

                Console.WriteLine("\nSystem has been trained.");
            }
            else
            {
                CNN_Testing(testingDigitImagesDatas, predictionIsOn, cnn, -1);
            }
        }

        public static void CNN_Testing(DigitImage[] digitImagesDatas, bool predictionIsOn, CNN cnn, int iterationCount)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            int testing_count = digitImagesDatas.Length;
            int correct_count = 0;
            double timeLimit = 0, error = 0;
            Matrix[] input, targets;

            Console.WriteLine("System is getting tested. You will see the results when it is done...\n");
            if (cursorTopTesting == 0)
                cursorTopTesting = Console.CursorTop;

            InitializeInputAndTarget(out input, out targets);
            timeLimit = stopwatch.ElapsedMilliseconds;

            for (int i = 0; i < testing_count; i++)
            {
                for (int j = 0; j < 28; j++)
                    for (int k = 0; k < 28; k++)
                        input[0][j, k] = digitImagesDatas[i].pixels[j][k];

                input[0].Normalize(0f, 255f, 0f, 1f);


                Matrix ans = null;
                if (predictionIsOn)
                {
                    ans = cnn.Predict(input);
                }
                else
                {
                    cnn.Train(input, targets[digitImagesDatas[i].label]);
                    ans = cnn.Layers[cnn.Layers.Length - 1].Output[0];
                }

                if (ans.GetMaxRowIndex() == digitImagesDatas[i].label)
                    correct_count++;

                if (stopwatch.ElapsedMilliseconds > timeLimit)
                {
                    // every 0.5 sec update error
                    timeLimit += 500;
                    error = cnn.GetError();
                }

                int val = Map(0, testing_count, 0, 100, i);
                ProgressBar(val, i, testing_count, error, stopwatch.ElapsedMilliseconds / 1000.0, cursorTopTesting);
            }
            double accuracy = (correct_count * 1f / testing_count) * 100.0;

            Console.WriteLine("\nIteration Count:" + iterationCount);
            Console.WriteLine("\nTime :" + (stopwatch.ElapsedMilliseconds / 1000.0).ToString("F4"));
            Console.WriteLine("\nAccuracy: %{0:F2}\n", accuracy);
            Console.WriteLine("Correct/All: {0}/{1}", correct_count, testing_count);

            cursorTopTesting = Console.CursorTop;

            if (accuracy >= 95 && iterationCount != -1)
            {
                string name = accuracy.ToString("F2") + "__" + iterationCount + "__";
                Random random = new Random();
                int length = 6;
                for (int i = 0; i < length; i++)
                {
                    char c = (char)random.Next('A', 'F' + 1);
                    name += c;
                }
                cnn.SaveData(name+".json");
            }
        }

        public static void CNN_OverfittingTest()
        {
            CNN cnn = new CNN();

            DigitImage[] digitImages = MNIST_Parser.ReadFromFile(DataSet.Testing, 100);

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            int iteration_count = 10;
            for (int i = 0; i < iteration_count; i++)
            {
                for(int j = 0; j < digitImages.Length; ++j)
                {
                    Matrix[] input = new Matrix[1];
                    input[0] = new Matrix(digitImages[j].pixels);

                    Matrix target = new Matrix(10, 1);
                    target[(int)digitImages[j].label, 0] = 1f;

                    cnn.Train(input, target);
                    double error = cnn.GetError();

                    int val = (int)((i - 0) / (double)(iteration_count - 1 - 0) * (100 - 0) + 0);
                    ProgressBar(val, i, iteration_count, error, stopwatch.ElapsedMilliseconds / 1000.0);
                }
                
            }

            for (int j = 0; j < digitImages.Length; ++j)
            {
                Matrix[] input = new Matrix[1];
                input[0] = new Matrix(digitImages[j].pixels);

                Matrix output = cnn.Predict(input);

                Console.WriteLine(output.ToString());
                Console.WriteLine(digitImages[j].ToString());
                Console.ReadLine();
            }
        }

        public static CNN CreateCNN(out int dialogResult)
        {
            string[] files = Directory.GetFiles(Path.Combine("..", "..", "CNN", "Configs"), "*.json");

            Console.WriteLine("0 > Train new network");
            for (int i = 0; i < files.Length; i++)
            {
                files[i] = files[i].Substring(files[i].LastIndexOf(Path.DirectorySeparatorChar) + 1);
                Console.WriteLine("{0} > {1}", i + 1, files[i]);
            }
            Console.Write(">> ");

            int indexOfFile = int.Parse(Console.ReadLine().Trim());
            dialogResult = indexOfFile;

            if (indexOfFile != 0 && indexOfFile <= files.Length)
            {
                return new CNN(files[indexOfFile - 1]);
            }
            else
                return new CNN();
        }

        public static void InitializeInputAndTarget(out Matrix[] input, out Matrix[] targets)
        {
            input = new Matrix[1];
            input[0] = new Matrix(28, 28);
            targets = new Matrix[10];

            for (int i = 0; i < 10; i++)
            {
                targets[i] = new Matrix(10, 1);
                for (int j = 0; j < 10; j++)
                {
                    targets[i][j, 0] = (i == j) ? 1.0f : 0.0f;
                }
            }
        }

        public static int Map(int oldMin, int oldMax, int newMin, int newMax, int current)
        {
            //Y = (X-A)/(B-A) * (D-C) + C
            return (int)((current - oldMin) / (double)(oldMax - 1 - oldMin) * (newMax - newMin) + newMin);
        }

        static void ProgressBar(int currentValue, int currentCount, int maxCount, double error, double timePassed = 0, int cursorTop = 0)
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

            Console.SetCursorPosition(pos + 1, cursorTop);
            Console.Write("#");
            Console.SetCursorPosition(14, cursorTop);
            Console.WriteLine(currentValue + "%");
            Console.SetCursorPosition(25, cursorTop);
            Console.WriteLine(currentCount + 1 + " / " + maxCount);
            Console.SetCursorPosition(45, cursorTop);
            Console.WriteLine("Time Passed: " + timePassed.ToString("F1"));
            Console.SetCursorPosition(70, cursorTop);
            Console.WriteLine("Error: {0:F13}", error);

            Console.CursorVisible = true;
        }
    }
}

