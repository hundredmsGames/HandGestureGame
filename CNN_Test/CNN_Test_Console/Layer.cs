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
        private Matrix input;
        private Matrix output;

        #region Get-Set

        public Matrix Input
        {
            get { return input; }
            protected set { input = value; }
        }

        public Matrix Output
        {
            get { return output; }
            protected set { output = value; }
        }

        #endregion
    }
}
