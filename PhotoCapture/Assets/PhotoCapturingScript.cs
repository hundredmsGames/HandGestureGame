using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//using UnityEngine.VR;
using UnityEngine.XR.WSA.WebCam;

using System.Linq;
using System;

public class PhotoCapturingScript : MonoBehaviour {

    public GameObject gameObject;
    public Material material;
    PhotoCapture photoCaptureObject = null;
    Texture2D targetTexture = null;

    // Use this for initialization
    void Start()
    {
        Resolution cameraResolution = PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First();
        targetTexture = new Texture2D(cameraResolution.width, cameraResolution.height);

        // Create a PhotoCapture object
        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObject) {
            photoCaptureObject = captureObject;
            CameraParameters cameraParameters = new CameraParameters( WebCamMode.VideoMode);
            
            cameraParameters.hologramOpacity = 0.0f;
            cameraParameters.cameraResolutionWidth = cameraResolution.width;
            cameraParameters.cameraResolutionHeight = cameraResolution.height;
            cameraParameters.pixelFormat = CapturePixelFormat.BGRA32;
            cameraParameters.frameRate = 30;
            // Activate the camera
            photoCaptureObject.StartPhotoModeAsync(cameraParameters, delegate (PhotoCapture.PhotoCaptureResult result) {
                // Take a picture
                photoCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);
                
            });
        });
        
    }
    private void Update()
    {
        if (photoCaptureObject != null)
            photoCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);
    }
    void OnCapturedPhotoToMemory(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        // Copy the raw image data into the target texture
        photoCaptureFrame.UploadImageDataToTexture(targetTexture);

        // Create a GameObject to which the texture can be applied
        GameObject quad = gameObject;
        Renderer quadRenderer = quad.GetComponent<Renderer>() as Renderer;
        quadRenderer.material = new Material(material);

        quad.transform.parent = this.transform;
        quad.transform.localPosition = new Vector3(0.0f, 0.0f, 3.0f);

        quadRenderer.material.SetTexture("_MainTex", targetTexture);

        // Deactivate the camera
        //photoCaptureObject.StopPhotoModeAsync(OnStoppedPhotoMode);
    }

    void OnStoppedPhotoMode(PhotoCapture.PhotoCaptureResult result)
    {
        // Shutdown the photo capture resource
        photoCaptureObject.Dispose();
        photoCaptureObject = null;
    }
    private void OnDisable()
    {
        photoCaptureObject.StopPhotoModeAsync(OnStoppedPhotoMode);
        Debug.Log("Stopped");
    }
}
