# Object Lens Camera Classifier

Mobile-first web app that opens the browser camera and classifies common objects with TensorFlow.js MobileNet.

## Run on this computer

From `D:\Solarcell\CAN_RS485\camera_classifier`:

```powershell
.\start_classifier.ps1
```

Or run the server command directly:

```powershell
python -m http.server 8090
```

Open Chrome or Edge:

```text
http://localhost:8090
```

Click `Start Camera` and allow camera access. The model loads from the TensorFlow.js CDN, so the browser needs internet access the first time.

## Use on a phone

Responsive layout is ready for phone screens, but mobile browsers usually allow camera access only on `localhost` or HTTPS. If you open the app from another computer over Wi-Fi using `http://192.168.x.x:8090`, the camera may be blocked.

For a real phone camera test, use one of these:

- Deploy this folder to an HTTPS web host.
- Use an HTTPS tunnel to your local server.
- Run a local server directly on the phone so the URL is `localhost`.

## Files

- `index.html` - app markup and TensorFlow.js scripts
- `styles.css` - responsive mobile/desktop layout
- `app.js` - camera handling, model loading, and live classification
- `manifest.webmanifest` - installable app metadata
