

## florence2_colab.ipynb

```
from transformers import AutoProcessor, AutoModelForCausalLM  
```

The Zero Shot Object Detection model is based on the pretrained model `microsoft/Florence-2-large`.

## sam_segmentation.ipynb

The SAM model `sam2_b` from ultralytics SAM is used for predicting on one image.

This took 300 seconds without GPU on one image.

As a result, bounding boxes and segmentation masks are calculated for all the objects detected in the image.

This is instance segmentation, e.g. each of the "windows" objects is a different class.
