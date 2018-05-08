using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.IO;
using System.ComponentModel;
using System.Windows.Forms;
using System.Drawing;
using System.Drawing.Imaging;

namespace CNN_Test
{
    public partial class MainForm : Form
    {
        private const string captureString = "Capture";
        private const string captureStopString = "Stop";

        private const int captureWidth  = 128;
        private const int captureHeight = 128;

        private static string desktopPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop));

        private Capture capture;
        private bool captureState;

        public MainForm()
        {
            InitializeComponent();
            
            InitializeCapture();
        }

        private void InitializeCapture()
        {
            capture = new Capture();
        }

        private void ToggleCaptureState()
        {
            if (captureState == false)
            {
                captureState = true;
                captureBackWorker.RunWorkerAsync();

                captureButton.Text = captureStopString;
            }
            else
            {
                captureState = false;
                capture.Pause();

                captureButton.Text = captureString;

            }
        }

        private void CaptureImage()
        {
            int counter = 0;
            while (captureState == true)
            {
                Image<Gray, byte> grayFrame = capture.QueryGrayFrame().Resize(captureWidth, captureHeight, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);

                captureBackWorker.WorkerReportsProgress = true;
                captureBackWorker.ReportProgress(0, grayFrame);
                captureBackWorker.WorkerReportsProgress = false;

                Bitmap bmp = grayFrame.Bitmap;
                bmp.Save(Path.Combine(desktopPath, "HandImages", (++counter) + ".png"), ImageFormat.Png);
            }
        }

        #region Component Events

        public void DoWork(object obj, DoWorkEventArgs e)
        {
            CaptureImage();
        }

        public void ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            pictureBox1.Image = ((Image<Gray, byte>)e.UserState).Bitmap;
            pictureBox1.Update();
        }

        private void captureButton_Click(object sender, EventArgs e)
        {
            ToggleCaptureState();
        }
        
        #endregion
    }
}
