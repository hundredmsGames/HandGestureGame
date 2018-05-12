using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MatrixLib;

namespace CNN_Test_Console
{
    class MaxPoolingLayer : Layer
    {
        private int kernel_size;
        private int stride;

        public MaxPoolingLayer(Matrix input, int kernel_size, int stride)
        {
            this.Input       = Input;
            this.Kernel_Size = kernel_size;
            this.Stride      = stride;
        }

        #region Get-Set

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

        #endregion
    }
}
