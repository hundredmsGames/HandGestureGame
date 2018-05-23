using MatrixLib;

namespace ConvNeuralNetwork
{
    class InputLayer : Layer
    {
        #region Variables

        private int width;
        private int height;
        private int channels;

        #endregion

        #region Constructors

        public InputLayer(int width, int height, int channels) : base(LayerType.INPUT)
        {
            this.width    = width;
            this.height   = height;
            this.channels = channels;
        }

        public override void Initialize()
        {
            base.Initialize();

            Input      = new Matrix[channels];
            Output     = new Matrix[channels];
            Output_d_E = new Matrix[channels];

            for (int i = 0; i < channels; i++)
            {
                Input[i]      = new Matrix(width, height);
                Output[i]     = new Matrix(width, height);
                Output_d_E[i] = new Matrix(width, height);
            }
        }

        #endregion

        #region Methods

        public override void FeedForward(Matrix[] input)
        {
            base.FeedForward(input);

            Input = input;
            Output = input;
            OutputLayer.Input = input;
        }

        #endregion

        #region Properties

        public int Width { get => width; set => width = value; }

        public int Height { get => height; set => height = value; }

        public int Channels { get => channels; set => channels = value; }

        #endregion
    }
}
