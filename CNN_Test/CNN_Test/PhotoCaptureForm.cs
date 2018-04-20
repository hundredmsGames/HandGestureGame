using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CNN_Test
{
    public partial class PhotoCaptureForm : Form
    {
        BackgroundWorker bw;

        public PhotoCaptureForm()
        {
            InitializeComponent();
        }

        public void DoWork(object obj, DoWorkEventArgs e)
        {
            Capture capture = new Capture();
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_WIDTH, 256);
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_HEIGHT, 256);

            while (true)
            {
                Image<Bgr, byte> img = capture.QueryFrame();
                Image<Gray, byte> grayFrame = img.Convert<Gray, byte>();

                for (int x = 0; x < grayFrame.Width; x++)
                {
                    for (int y = 0; y < grayFrame.Height; y++)
                    {
                        int b = (int)grayFrame.Data[y, x, 0];
                        // int g = (int)grayFrame.Data[x, y, 1];
                        //	int r = (int)grayFrame.Data[x, y, 2];

                        if (IsPixelSkin(b, b, b) == true)
                        {
                            grayFrame.Data[y, x, 0] = 0;
                            //	grayFrame.Data[y, x, 1] = 0;
                            //	grayFrame.Data[y, x, 2] = 0;
                        }
                    }
                }


                bw.WorkerReportsProgress = true;
                bw.ReportProgress(0, grayFrame);

                System.Threading.Thread.Sleep(100);
            }
        }

        public void ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            pictureBox1.Image = ((Image<Gray, byte>)e.UserState).Bitmap;
            pictureBox1.Update();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            bw = new BackgroundWorker();
            bw.DoWork += new DoWorkEventHandler(DoWork);
            bw.ProgressChanged += new ProgressChangedEventHandler(ProgressChanged);

            bw.RunWorkerAsync();
        }

        public static bool IsPixelSkin(int r, int g, int b)
        {
            if (r >= 120 && r <= 140)
                return true;
            else
                return false;
        }
    }
}
