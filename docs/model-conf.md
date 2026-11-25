# model.conf: a model's own detection settings

Every export (`train/build_model.py`, `train/qat/train.py` `model_conf()`) and every model in the
library ships `model.conf` next to the model file. It holds the facts a camera needs to run the
model, in rvd's own `[motion]` keys, so nobody has to type them into raptor.conf:

```
# thingino-ai model sidecar: rvd [motion] defaults for this model
# (keys in raptor.conf [motion] override these)
[motion]
algorithm = yolo
num_classes = 4
class_names = person,pet,package,vehicle
class_thresholds = 40,30,30,40
anchors_16 = 10,14,23,27,37,58
anchors_32 = 81,82,135,169,344,319
input = 416x320
preprocess = stretch
```

| Key | Meaning |
|---|---|
| `algorithm` | always `yolo` for these models |
| `num_classes`, `class_names` | the model's classes, in output order |
| `class_thresholds` | suggested per-class score thresholds in % (person and vehicle 40, others 30, from the live tests) |
| `anchors_<stride>` | (w, h) x 3 in model-input pixels, one line per head stride (persondet-type: 16 and 32; YOLO-type: 8, 16 and 32). A venus `.bin` carries no anchors, so they must come from here |
| `input` | the model's input size, W x H |
| `preprocess` | how a frame becomes the input: `stretch` (the 16:9 frame resized to the input, as in training) |

**Lookup** (rvd, raptor branch `rvd-detect-classes`): for `[motion] model = /path/X.bin`, rvd reads
`/path/X.conf`, else `/path/X.bin.conf`, and uses its `[motion]` keys as defaults; any key set in
raptor.conf `[motion]` wins. Camera policy (which classes alert, record or notify) stays in
raptor.conf. raptor-hal's onna backend takes the anchors rvd passes (`param->anchors`).
