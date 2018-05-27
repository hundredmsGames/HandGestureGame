using System;
using MatrixLib;

namespace ConvNeuralNetwork
{
    class ConvLayer : Layer
    {
        #region Variables

        private int filters;
        private int kernel_size;
        private int stride;
        private int padding;
        private Matrix[,] kernels;

        Func<float, float> activation;
        Func<float, float> derOfActivation;

        #endregion

        #region Constructors

        public ConvLayer(int filters, int kernel_size, int stride, ActivationType activationType, int padding = -1) : base(LayerType.CONVOLUTIONAL)
        {
            this.Filters = filters;
            this.Kernel_Size = kernel_size;
            this.Stride = stride;
            this.Padding = padding;
            
            // Padding
            if(padding == -1)
            {
                // Padding is 0 by default.
                padding = 0;
            }

            var activations = ActivationFunctions.GetActivationFuncs(activationType);
            activation = activations.Item1;
            derOfActivation = activations.Item2;
        }

        public override void Initialize()
        {
            base.Initialize();

            // Initialize output
            this.Input = InputLayer.Output;
            this.Output = new Matrix[filters];
            this.Output_d_E = new Matrix[filters];

            int in_r = this.Input[0].rows;
            int in_c = this.Input[0].cols;
            int f = Kernel_Size;
            int p = Padding;
            int s = Stride;

            int out_size_r = (in_r - f + 2 * p) / s + 1;
            int out_size_c = (in_c - f + 2 * p) / s + 1;
            for (int i = 0; i < filters; i++)
            {
                this.Output[i] = new Matrix(out_size_r, out_size_c);
                this.Output_d_E[i] = new Matrix(out_size_r, out_size_c);
            }

            // Initialize kernel
            kernels = new Matrix[filters, Input.Length];
            for (int i = 0; i < filters; i++)
            {
                for (int j = 0; j < Input.Length; j++)
                {
                    kernels[i, j] = new Matrix(kernel_size, kernel_size);
                    kernels[i, j].Randomize();
                }
            }
        }

        #endregion

        #region Methods

        public override void FeedForward()
        {
            base.FeedForward();

            int out_idx_r, out_idx_c;

            for (int fil_idx = 0; fil_idx < Filters; fil_idx++)
            {
                //reset values
                Output[fil_idx].FillZero();

                out_idx_r = 0;
                for (int r = 0; r < Input[0].rows && out_idx_r < Output[0].rows; r += stride, out_idx_r++)
                {
                    out_idx_c = 0;
                    for (int c = 0; c < Input[0].cols && out_idx_c < Output[0].cols; c += stride, out_idx_c++)
                    {
                        for (int ch = 0; ch < Input.Length; ch++)
                        {
                            for (int i = 0; i < kernel_size; i++)
                            {
                                for (int j = 0; j < kernel_size; j++)
                                {
                                    Output[fil_idx][out_idx_r, out_idx_c] += Input[ch][r + i, c + j] * kernels[fil_idx, ch][i, j];
                                }
                            }
                        }

                        Output[fil_idx][out_idx_r, out_idx_c] = activation(Output[fil_idx][out_idx_r, out_idx_c]);
                    }
                }
            }

            this.OutputLayer.Input = Output;
        }

        public override void Backpropagation()
        {
            base.Backpropagation();

            Matrix kernel_d_E = new Matrix(kernel_size, kernel_size);

            for (int fil_idx = 0; fil_idx < Filters; fil_idx++)
            {
                for (int ch = 0; ch < Input.Length; ch++)
                {
                    //reset values
                    InputLayer.Output_d_E[ch].FillZero();
                    kernel_d_E.FillZero();

                    for (int i = 0, r = 0; r < Output_d_E[fil_idx].rows && i < Input[ch].rows; i += stride, r++)
                    {
                        for (int j = 0, c = 0; c < Output_d_E[fil_idx].cols && j < Input[ch].cols; j += stride, c++)
                        {
                            float derOfAct = derOfActivation(Output[fil_idx][r, c]);

                            for (int p = 0; p < kernel_size; p++)
                            {
                                for (int q = 0; q < kernel_size; q++)
                                {
                                    kernel_d_E[p, q] += derOfAct * Output_d_E[fil_idx][r, c] * Input[ch][i + p, j + q];

                                    if (LayerIndex != 1)
                                    {
                                        InputLayer.Output_d_E[ch][i + p, j + q] +=
                                            kernels[fil_idx, ch][p, q] * derOfAct * Output_d_E[fil_idx][r, c];
                                    }
                                }
                            }               
                        }
                    }

                    kernels[fil_idx, ch] = kernels[fil_idx, ch] - (Network.LearningRate * kernel_d_E);
                }
            }
        }

        #endregion

        #region Properties

        public int Filters
        {
            get { return filters; }
            protected set { filters = value; }
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

        public Matrix[,] Kernels
        {
            get { return kernels; }
             set { kernels = value; }
        }

        public int Padding { get => padding; set => padding = value; }

        #endregion
    }
}
