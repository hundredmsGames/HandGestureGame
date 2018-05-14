using MatrixLib;

namespace ConvNeuralNetwork
{
    class Layer
    {
        private Matrix input;
        private Matrix output;

        private Layer inputLayer;
        private Layer outputLayer;

        private LayerType layerType;
        private int layerIndex;
        public CNN network;

        #region Constructor

        public Layer(LayerType layerType)
        {
            this.layerType = layerType;
        }
      
       
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


        public Layer OutputLayer
        {
            get { return outputLayer; }
            set { outputLayer = value; }
        }



        public Layer InputLayer
        {
            get { return inputLayer; }
            set { inputLayer = value; }
        }

        internal CNN Network { get => network; set => network = value; }
        internal LayerType LayerType { get => layerType; set => layerType = value; }
        public int LayerIndex { get => layerIndex; set => layerIndex = value; }



        #endregion
    }
}
