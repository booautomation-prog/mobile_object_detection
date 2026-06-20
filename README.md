# YOLO Model Folder

Put your YOLOv8 ONNX detection model here:

```text
models/yolov8n.onnx
```

Recommended export:

```powershell
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
```

Then rename or copy the exported file to:

```text
yolov8n.onnx
```

Upload this `models` folder to GitHub Pages together with `index.html`, `app.js`, and `styles.css`.
