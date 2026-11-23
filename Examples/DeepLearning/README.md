# Deep learning object detection

The **Object Detection (Deep Learning, ONNX)** component (`ObjectDetectionDNN`, in the *Particle detection* category) runs an object detector exported to [ONNX](https://onnx.ai) with OpenCV's `dnn` module. Each detection becomes a particle with a center, a bounding box, an area (box width × height) and a confidence score, which a tracker can then link into tracks.

`detector.swistrack` is an example pipeline:

```
InputFileAVI -> ConvertInputToColor -> ObjectDetectionDNN -> MultiObjectTracking -> OutputTracksToFiles
```

To use it, put your video (`video.avi`), your model (`best.onnx`) and its class names (`classes.txt`) next to the pipeline file, or change the paths in the GUI, then run it:

```sh
SwisTrack-batch detector.swistrack
```

## Try it: tracking pedestrians

`people.swistrack` tracks the pedestrians in OpenCV's sample video `vtest.avi` with the pretrained YOLOX-Nano model (Apache-2.0; COCO class 0 = person). Download both into this folder and run:

```sh
curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
curl -LO https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi
SwisTrack-batch people.swistrack        # writes testing/people.csv (frame,id,x,y,vx,vy)
```

On an Apple M-series CPU, the 795 frames take about 6 s in total (about 8 ms per inference at 416 × 416), producing 24 tracks of 214 frames on average, with about 6.5 people tracked per frame. The detections match an independent Python implementation of the YOLOX post-processing exactly.

SwisTrack needs OpenCV with the `dnn` module, which the usual packages include (Ubuntu `libopencv-dev`, Homebrew `opencv`, vcpkg `opencv4`). If your OpenCV was built without it, the component still shows up but reports an error.

## Supported models

| Model family | Output | License of the training code |
|---|---|---|
| YOLOv8, YOLO11 ([Ultralytics](https://github.com/ultralytics/ultralytics)) | `1 x (4+C) x N`, no objectness | AGPL-3.0 (or a commercial license) |
| [YOLOv5](https://github.com/ultralytics/yolov5) | `1 x N x (5+C)` with objectness | AGPL-3.0 |
| [YOLOv7](https://github.com/WongKinYiu/yolov7) | `1 x N x (5+C)` with objectness | GPL-3.0 |
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | `1 x N x (5+C)`, raw grid outputs for strides 8/16/32 | Apache-2.0 |

`C` is the number of classes, and `N` the number of candidate boxes. With *Model family* set to *Auto*, the component works out the family from the output shape when it loads the model (the status shows what it found). An output with the attributes first is YOLOv8/YOLO11. An output with 3 boxes per grid cell is YOLOv5/YOLOv7. An output with 1 box per grid cell is taken to be YOLOX. Pick the family explicitly if the guess is wrong.

Note that the license of the training code may apply to the trained weights, and to software that ships them. If AGPL-3.0 is a problem, use YOLOX (Apache-2.0). RT-DETR (Apache-2.0, [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)) is another option for the future, but its output format is not supported by this component yet.

## Training and exporting a model

Label your images with bounding boxes, for example with [CVAT](https://www.cvat.ai) or [Label Studio](https://labelstud.io), and export them in the YOLO format. A few hundred labeled frames taken from your own setup (same camera, lighting and animals or robots) usually work better than a large generic dataset.

### YOLOv8 / YOLO11 (Ultralytics)

```sh
pip install ultralytics
yolo detect train data=data.yaml model=yolo11n.pt imgsz=640 epochs=100
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12 imgsz=640
```

Export without `nms=True`: this component does its own non-maximum suppression and expects the raw `1 x (4+C) x N` output. `opset=12` works with older OpenCV versions (such as 4.5 on Ubuntu 22.04). To get the class names, write the `names` from `data.yaml` to `classes.txt`, one per line, in class order.

### YOLOv5

```sh
python train.py --data data.yaml --weights yolov5s.pt --img 640 --epochs 100
python export.py --weights runs/train/exp/weights/best.pt --include onnx --opset 12 --img 640
```

### YOLOv7

```sh
python export.py --weights best.pt --grid --simplify --img-size 640 640
```

`--grid` is needed so that the boxes are decoded inside the model. Do not use `--end2end`, which adds NMS to the model.

### YOLOX

```sh
python tools/train.py -f exps/example/custom/yolox_s.py -d 1 -b 16 --fp16 -o -c yolox_s.pth
python tools/export_onnx.py --output-name best.onnx -f exps/example/custom/yolox_s.py -c YOLOX_outputs/yolox_s/best_ckpt.pth
```

Export without `--decode_in_inference`, because the component decodes the YOLOX grid itself. YOLOX models expect BGR images with pixel values from 0 to 255, placed in the top-left corner of the padded input. With *Channel order* and *Pixel values* on *Auto*, the component does this when the model family is YOLOX. Models from early YOLOX releases (those that need the `--legacy` option in the YOLOX demos) expect RGB images normalized with the ImageNet mean and standard deviation, which this component does not support.

## Input size

The input width and height must match the size the model was exported with (`imgsz`), unless the model was exported with dynamic axes (`dynamic=True`). Both must be multiples of 32. The image is scaled to fit (keeping its aspect ratio) and padded with gray ("letterbox"), and the boxes are mapped back to the original image.

- **640 × 640** is the default for most models, and a good start.
- **320 or 416** is 2–4 times faster on a CPU, and enough for large objects.
- **960 or 1280** helps with small objects (for example insects or fish in a wide arena), but is much slower. Train with the same size.
- For wide video, a non-square size such as `imgsz=[384,640]` (height, width) avoids wasting time on padding. Set *Input width* to 640 and *Input height* to 384 in that case.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| Model (ONNX) | | The `.onnx` file. Relative paths are relative to the folder of the saved pipeline file. |
| Model family | Auto | Output format (see above). |
| Input width, height | 640 × 640 | Network input size. |
| Channel order, Pixel values | Auto | Preprocessing: RGB and 0..1 for Ultralytics/YOLOv5/YOLOv7, BGR and 0..255 for YOLOX. |
| Run on | OpenCV (CPU) | OpenCL or CUDA, if your OpenCV build supports them. Otherwise the component falls back to the CPU with a warning. |
| Min. score | 0.1 | Detections with a lower score are dropped. This is lower than the usual 0.25, because trackers like ByteTrack use weak detections to keep tracks alive through occlusions. The tracker decides which detections count as strong. |
| NMS IoU threshold | 0.45 | Overlapping boxes with an intersection over union above this are suppressed, keeping the one with the higher score. |
| NMS per class | on | Only suppress boxes of the same class. Turn this off if the same object is sometimes detected as different classes. |
| Max. detections | 300 | Maximum number of detections per frame (0 = no limit). |
| Classes | (all) | Class numbers to keep, separated by commas, for example `0, 2`. |
| Class names | | Text file with one class name per line, used for display. |
| Highlight score (display) | 0.5 | Boxes with at least this score are drawn thick and green, weaker ones thin and orange. This only affects the display. |

The class of a detection is not stored in the particle. To track a single class, use *Classes*.

## Small objects: tiled inference

The model sees the image scaled down to its input size, so objects that are only a few pixels large after scaling are missed. With *Tile columns/rows* > 1, the image is processed in overlapping tiles at full model resolution (as in SAHI, slicing aided hyper inference). Boxes cut off at an inner tile border are dropped (the object is fully visible in the neighboring tile, as long as it is smaller than the overlap), the full image is processed as well for large objects (*Also process the full image*), and all detections are merged by non-maximum suppression. Each tile costs one inference: 2 × 2 tiles plus the full image is 5 times slower. Use tiling only when objects are small compared to the image (e.g. many small animals in a large arena); for the pedestrians in `vtest.avi`, it does not find more people.

## Performance

The model loads, and runs once on a blank image, when the pipeline starts or when a model parameter changes. The status then shows the duration of that first inference. Changing a threshold takes effect on the next frame without reloading the model. The step duration of the component shows the time per frame.

On a CPU, small models (YOLOv8n, YOLO11n, YOLOX-Nano or YOLOX-Tiny) at 640 × 640 typically take a few tens of milliseconds per frame. Larger models (s, m, l) and input sizes are much slower. If that is too slow:

- Use a smaller model or input size.
- Restrict the classes (e.g. `0` for persons in COCO models).
- Use a GPU. This needs an OpenCV built with CUDA (not the standard packages), or OpenCL. OpenCV 5's new inference engine currently runs on the CPU only.
