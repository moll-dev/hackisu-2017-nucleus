using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading;
using System.Threading.Tasks;
using Windows.Devices.Enumeration;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.Graphics.Imaging;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.Devices;
using Windows.Media.MediaProperties;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.System.Display;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace KinectCapture
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        MediaCapture mediaCapture;
        private readonly DisplayRequest displayRequest = new DisplayRequest();

        int captureIdx;

        bool isRecording;
        private StorageFolder captureFolder = null;

        private AdvancedPhotoCapture advancedCapture;

        MediaFrameReader colorFrameReader;
        MediaFrameReader depthFrameReader;


        Mutex mutex;

        DateTime prev;


        public MainPage()
        {
            this.InitializeComponent();

            mutex = new Mutex();
            captureIdx = 0;

            isRecording = false;
            
        }

        protected override async void OnNavigatedTo(NavigationEventArgs e)
        {

            await InitializeCameraAsync();

            foreach (string source in mediaCapture.FrameSources.Keys)
            {
                Debug.WriteLine(source);
            }

            displayRequest.RequestActive();
            PreviewBox.Source = mediaCapture;
            
            await mediaCapture.StartPreviewAsync();
            

            string date = DateTime.Now.ToString("HHmmss");

            var picturesLibrary = await StorageLibrary.GetLibraryAsync(KnownLibraryId.Pictures);
            captureFolder = picturesLibrary.SaveFolder ?? ApplicationData.Current.LocalFolder;

            captureFolder = await captureFolder.CreateFolderAsync(string.Format("Kinect_Capture_{0}", date));

        }

        #region Capture Setup
        private async Task<DeviceInformation> FindKinect()
        {
            Debug.WriteLine("EnumerateCameras");

            var devices = await DeviceInformation.FindAllAsync(DeviceClass.VideoCapture);

            foreach (var device in devices)
            {
                if (device.Name.Contains("Kinect"))
                {
                    return device;
                }
            }

            return null;
        }

        private async Task InitializeCameraAsync()
        {
            Debug.WriteLine("InitializeCameraAsync");
            
            // Grab that Kinect

            DeviceInformation cameraDevice = await FindKinect();

            Debug.WriteLine("Camera Device ID: " + cameraDevice.Id);
            
            if (cameraDevice == null)
            {
                Debug.WriteLine("No camera device found!");
                return;
            }

            mediaCapture = new MediaCapture();
            //mediaCapture.Failed += MediaCapture_Failed;

            var settings = new MediaCaptureInitializationSettings {
                SharingMode = MediaCaptureSharingMode.SharedReadOnly,
                VideoDeviceId = cameraDevice.Id,
                MemoryPreference = MediaCaptureMemoryPreference.Cpu
            };

            // Initialize MediaCapture
            try
            {
                await mediaCapture.InitializeAsync(settings);
            }
            catch (UnauthorizedAccessException)
            {
                Debug.WriteLine("The app was denied access to the camera");
            }

            var sourceGroup = await MediaFrameSourceGroup.FromIdAsync(cameraDevice.Id);
            MediaFrameSourceInfo colorSourceInfo = sourceGroup.SourceInfos.Where(si => si.SourceKind == MediaFrameSourceKind.Color).First();
            MediaFrameSourceInfo depthSourceInfo = sourceGroup.SourceInfos.Where(si => si.SourceKind == MediaFrameSourceKind.Depth).First();

            MediaFrameSource colorSource;
            MediaFrameSource depthSource;

            mediaCapture.FrameSources.TryGetValue(colorSourceInfo.Id, out colorSource);
            mediaCapture.FrameSources.TryGetValue(depthSourceInfo.Id, out depthSource);

            colorFrameReader = await mediaCapture.CreateFrameReaderAsync(colorSource);
            depthFrameReader = await mediaCapture.CreateFrameReaderAsync(depthSource);

            colorFrameReader.FrameArrived += ColorFrameReader_FrameArrived;

            await depthFrameReader.StartAsync();
            await colorFrameReader.StartAsync();
        }

        private async Task SaveFrames(SoftwareBitmap colorBitmap, SoftwareBitmap depthBitmap)
        {
            
            StorageFile outputColorFile = await captureFolder.CreateFileAsync(string.Format("capture{0}c.jpg", captureIdx), CreationCollisionOption.ReplaceExisting);

            using (IRandomAccessStream stream = await outputColorFile.OpenAsync(FileAccessMode.ReadWrite))
            {
                BitmapEncoder encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.JpegEncoderId, stream);
                encoder.SetSoftwareBitmap(colorBitmap);
                encoder.IsThumbnailGenerated = true;

                try
                {
                    await encoder.FlushAsync();
                }
                catch (Exception err)
                {
                    Debug.WriteLine("Error saving color image");
                }

                if (encoder.IsThumbnailGenerated == false)
                {
                    await encoder.FlushAsync();
                }
            }

            StorageFile outputDepthFile = await captureFolder.CreateFileAsync(string.Format("capture{0}d.jpg", captureIdx), CreationCollisionOption.ReplaceExisting);

            using (IRandomAccessStream stream = await outputDepthFile.OpenAsync(FileAccessMode.ReadWrite))
            {
                BitmapEncoder encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.JpegEncoderId, stream);
                encoder.SetSoftwareBitmap(depthBitmap);
                encoder.IsThumbnailGenerated = true;

                try
                {
                    await encoder.FlushAsync();
                }
                catch (Exception err)
                {
                    Debug.WriteLine("Error saving depth image");
                }

                if (encoder.IsThumbnailGenerated == false)
                {
                    await encoder.FlushAsync();
                }
            }
        }

        private void ColorFrameReader_FrameArrived(MediaFrameReader sender, MediaFrameArrivedEventArgs args)
        {
            using (var colorFrame = sender.TryAcquireLatestFrame())
            {
                using (var depthFrame = depthFrameReader.TryAcquireLatestFrame())
                {
                    if (colorFrame != null && depthFrame != null)
                    {
                        DateTime now = DateTime.Now;

                        mutex.WaitOne(-1);

                        if ((now - prev).TotalMilliseconds < 100)
                        {
                            mutex.ReleaseMutex();
                            return;
                        }
                        

                        if (isRecording)
                        {
                            Task t = SaveFrames(
                                FrameRenderer.ConvertToDisplayableImage(colorFrame.VideoMediaFrame), 
                                FrameRenderer.ConvertToDisplayableImage(depthFrame.VideoMediaFrame));

                            t.Wait();

                            captureIdx += 1;
                        }

                        prev = now;
                        mutex.ReleaseMutex();
                        
                        
                    }
                }
                
            }
        }



        #endregion
        

        private void RecordButton_Click(object sender, RoutedEventArgs e)
        {
            isRecording = !isRecording;
        }

    }

}
