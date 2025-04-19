

## florence2_colab.ipynb

```
from transformers import AutoProcessor, AutoModelForCausalLM
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

The pretrained model `microsoft/Florence-2-large` is used.

The inputs are an image, a task prompt (command) and optionally a text prompt. 

Depending on the command, the output can be a text or a set of bounding boxes (rectangular or polygonal) and labels.

The command `<MORE_DETAILED_CAPTION>` produces a detailed text:

  > The image shows a two-story house with a red tiled roof. The house is painted in a light beige color and has a white exterior. There are three windows on the front of the house, each with white frames. The windows have black shutters and there is a small balcony on the second floor. A blue parking sign is attached to a pole on the right side of the image. A silver car is parked on the street next to the house. The sky is overcast and the ground is wet, suggesting that it has recently rained.


## sam_segmentation.ipynb

```
from ultralytics import SAM
model = SAM("sam2_b.pt")
results = model("autoima/279_1440x960.jpg", project="segment_everything", save=True, save_txt=True)
```

The SAM model `sam2_b` from ultralytics SAM is used for predicting all segments in one image.

This approach was not really useful for the Parsing House Facades project:
- computationally intensive, it took 300 seconds without GPU on one image
- too many results as we are only interested in selected objects. Bounding boxes and segmentation masks are calculated for all the objects detected in the image.
- this is instance segmentation, e.g. each of the "windows" objects is considered a different class. So there is no notion of a "windows" class.

![result of SAM segment everything](../assets/sam_segment_everything.JPG "result of SAM segment everything")


