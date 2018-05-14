using MatrixLib;

namespace ConvNeuralNetwork
{
    class Layer
    {
        private Matrix input;
        private Matrix output;

        private Layer prevLayer;
        private Layer nextLayer;

        private Layer network;



        #region Constructor

      
       
        #endregion

        #region Methods
        virtual public void FeedForward()
        {

        }
        virtual public void Backpropagation()
        {

        }
        #endregion
        #region Properties

        public Matrix Input
        {
            get { return input; }
            set { input = value; }
        }

        public Matrix Output
        {
            get { return output; }
            set { output = value; }
        }


        public Layer NextLayer
        {
            get { return nextLayer; }
            set { nextLayer = value; }
        }



        public Layer PrevLayer
        {
            get { return prevLayer; }
            set { prevLayer = value; }
        }

        internal Layer Network { get => network; set => network = value; }



        #endregion
    }
}
