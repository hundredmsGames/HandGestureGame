using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MatrixLib;
namespace CNN_Test_Console
{
    class Layer
    {
        private Matrix kernel;

        public Matrix Kernel
        {
            get { return kernel; }
            protected set { kernel = value; }
        }

        private Matrix output;

        public Matrix Output
        {
            get { return output; }
            protected set { output = value; }
        }

        private Matrix input;

        public Matrix Input
        {
            get { return input; }
            protected set { input = value; }
        }
        private int kernel_size;

        public int Kernel_Size
        {
            get { return kernel_size; }
            protected set { kernel_size = value; }
        }

        private int stride;

        public int Stride
        {
            get { return stride; }
           protected set { stride = value; }
        }

        public Layer(int _kernel_size,int _stride,Matrix input)
        {
            //initialize all matrices

            kernel = new Matrix(kernel_size, kernel_size);
           
        }

    }
}
