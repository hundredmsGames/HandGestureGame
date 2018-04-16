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

    public string deviceName;
    WebCamTexture wct;

    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        deviceName = devices[1].name;
        wct = new WebCamTexture(deviceName, 400, 300, 12);
        GetComponent<Renderer>().material.mainTexture = wct;
        wct.Play();
    }

    void TakeSnapshot()
    {
        Texture2D snap = new Texture2D(wct.width, wct.height);
        snap.SetPixels(wct.GetPixels());
        snap.Apply();
    }

    void OnGUI()
    {
        if (GUI.Button(new Rect(10, 70, 50, 30), "Click"))
            TakeSnapshot();


        // Create a GameObject to which the texture can be applied
        GameObject quad = gameObject;
        Renderer quadRenderer = quad.GetComponent<Renderer>() as Renderer;
        quadRenderer.material = new Material(material);

        quad.transform.parent = this.transform;
        quad.transform.localPosition = new Vector3(0.0f, 0.0f, 3.0f);

        quadRenderer.material.SetTexture("_MainTex", wct);

    }

    /*
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
    */
}
