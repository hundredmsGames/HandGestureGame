using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ConvNeuralNetwork;

namespace CNN_Test_Console
{
    class Program
    {
        static void Main(string[] args)
        {

            CNN cnn = new CNN();
            cnn.Train();

            Console.ReadLine();
        }
    }
}
