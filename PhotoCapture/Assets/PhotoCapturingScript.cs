using UnityEngine;

public class PhotoCapturingScript : MonoBehaviour {

    public GameObject gameObject;
    public Material material;
    int deviceIndex = 1;

    public string deviceName;
    WebCamTexture wct;

    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        deviceName = devices[deviceIndex].name;
        wct = new WebCamTexture(deviceName, 400, 300, 12);
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
        if (GUI.Button(new Rect(10, 70, 70, 30), "Click"))
            TakeSnapshot();

        if (GUI.Button(new Rect(10, 100, 70, 30), "Change"))
        {
            WebCamDevice[] devices = WebCamTexture.devices;
            wct.Stop();
            deviceIndex = (++deviceIndex) % 2;

            deviceName = devices[deviceIndex].name;
            wct = new WebCamTexture(deviceName, 500, 500, 12);
            GetComponent<Renderer>().material.mainTexture = wct;
            wct.Play();
        }



        // Create a GameObject to which the texture can be applied
        GameObject quad = gameObject;
        Renderer quadRenderer = quad.GetComponent<Renderer>() as Renderer;
        quadRenderer.material = new Material(material);

        quad.transform.parent = this.transform;
        quad.transform.localPosition = new Vector3(0.0f, 0.0f, 3.0f);

        quadRenderer.material.SetTexture("_MainTex", wct);

    }
}
