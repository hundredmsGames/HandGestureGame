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

        private const int captureWidth = 128;
        private const int captureHeight = 128;

        private static string handImagesPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
            "HandImages");

        private Capture capture;
        private bool captureState;

        public MainForm()
        {
            InitializeComponent();

            //BitmapToIdx();
            //Read();

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
                bmp.Save(Path.Combine(handImagesPath, "Gesture", "1_" + (++counter) + ".png"), ImageFormat.Png);
            }
        }

        private void BitmapToIdx()
        {
            string[] images = Directory.GetFiles(Path.Combine(handImagesPath, "Gesture"));

            FileStream writeStream1 = new FileStream(Path.Combine(handImagesPath, "images.idx"), FileMode.Append);
            FileStream writeStream2 = new FileStream(Path.Combine(handImagesPath, "labels.idx"), FileMode.Append);
            {
                BinaryWriter writeBinary1 = new BinaryWriter(writeStream1);
                BinaryWriter writeBinary2 = new BinaryWriter(writeStream2);

                Bitmap firstImage = new Bitmap(images[0]);

                int magic = 126;
                int numImages = images.Length;
                int rows = firstImage.Width;
                int cols = firstImage.Height;

                writeBinary1.Write(magic);
                writeBinary1.Write(numImages);
                writeBinary1.Write(rows);
                writeBinary1.Write(cols);

                writeBinary2.Write(magic);
                writeBinary2.Write(numImages);

                foreach (string image in images)
                {
                    Bitmap bmp = new Bitmap(image);

                    int label = image[image.LastIndexOf('\\') + 1] - '0';
                    writeBinary2.Write((byte)label);

                    for (int i = 0; i < rows; i++)
                    {
                        for (int j = 0; j < cols; j++)
                        {
                            writeBinary1.Write(bmp.GetPixel(j, i).G);
                        }
                    }
                }

                writeBinary1.Close();
                writeBinary2.Close();
            }
        }

        #region This is for test purpose.
        private void Read()
        {
            using (FileStream readStream = new FileStream(Path.Combine(handImagesPath, "images.idx"), FileMode.Open))
            {
                BinaryReader reader = new BinaryReader(readStream);

                int magic1 = reader.ReadInt32();
                int numImages = reader.ReadInt32();
                int numRows = reader.ReadInt32();
                int numCols = reader.ReadInt32();

                Console.WriteLine(magic1);
                Console.WriteLine(numImages);
                Console.WriteLine(numRows);
                Console.WriteLine(numCols);

                for (int j = 0; j < numImages; j++)
                {
                    Console.WriteLine(reader.ReadByte());
                    for (int i = 0; i < 128 * 128 - 1; i++)
                        reader.ReadByte();
                }

                reader.Close();
            }
        }
        #endregion

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
