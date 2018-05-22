using MatrixLib;

namespace ConvNeuralNetwork
{
    class Layer
    {
        #region Variables

        private Matrix[] input;
        private Matrix[] output;

        // Derivative of error with respect to output
        private Matrix[] output_d_E;

        private Layer inputLayer;
        private Layer outputLayer;

        private LayerType layerType;
        private int layerIndex;
        private CNN network;
        
        #endregion

        #region Constructors

        public Layer(LayerType layerType)
        {
            this.layerType = layerType;
        }

        #endregion

        #region Methods

        virtual public void FeedForward()
        {
    
        }
        virtual public void FeedForward(Matrix input)
        {

        }
        virtual public void FeedForward(Matrix[] input)
        {

        }
        virtual public void Backpropagation()
        {

        }
        virtual public void Initialize()
        {
            //if(InputLayer != null)
            //    this.Input = InputLayer.Output;
            switch (this.layerType)
            {
                case LayerType.INPUT:
                    Output_d_E = new Matrix[1];
                    break;
                case LayerType.CONVOLUTIONAL:
                    Output_d_E = new Matrix[(this as ConvLayer).Filters];
                    break;
                case LayerType.MAXPOOLING:
                    Output_d_E = new Matrix[1];
                    break;
                case LayerType.FULLY_CONNECTED:
                    Output_d_E = new Matrix[1];
                    break;
                default:
                    break;
            }
        }
        
        #endregion

        #region Properties

        public Matrix[] Input
        {
            get { return input; }
            set { input = value; }
        }

        public Matrix[] Output
        {
            get { return output; }
            set { output = value; }
        }

        public Matrix[] Output_d_E
        {
            get { return output_d_E; }
            set { output_d_E = value; }
        }

        public Layer InputLayer
        {
            get { return inputLayer; }
            set { inputLayer = value; }
        }

        public Layer OutputLayer
        {
            get { return outputLayer; }
            set { outputLayer = value; }
        }

        public CNN Network { get => network; set => network = value; }

        public LayerType LayerType { get => layerType; set => layerType = value; }

        public int LayerIndex { get => layerIndex; set => layerIndex = value; }

        #endregion
    }
}
