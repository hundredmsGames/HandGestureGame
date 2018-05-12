using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MatrixLib;

namespace CNN_Test_Console
{
    class ConvLayer : Layer
    {
        private int filters;
        private int kernel_size;
        private int stride;
        private Matrix kernel;

        public ConvLayer(Matrix input, int filters, int kernel_size, int stride)
        {
            this.Input       = input;
            this.Filters     = filters;
            this.Kernel_Size = kernel_size;
            this.Stride      = stride;

            // Initialize kernel
            kernel = new Matrix(kernel_size, kernel_size);
            kernel.Randomize();
        }

        #region Get-Set

        public int Filters
        {
            get { return filters; }
            protected set { filters = value }
        }

        public int Kernel_Size
        {
            get { return kernel_size; }
            protected set { kernel_size = value; }
        }

        public int Stride
        {
            get { return stride; }
            protected set { stride = value; }
        }

        public Matrix Kernel
        {
            get { return kernel; }
            protected set { kernel = value; }
        }

        #endregion
    }
}
