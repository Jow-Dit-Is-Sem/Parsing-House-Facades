## groundingdino_vanilla.ipynb

```
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
```

The Zero Shot Object Detection model is based on the pretrained model `IDEA-Research/grounding-dino-base`.

This model is ran for each of the 181 images in the `ds/grounding_dino_vanilla` directory. 
This is a relatively fast process even without GPU, taking 30 seconds per image.

Bounding boxes are converted to YOLO format labels, and together with the original images and a hand crafted YAML file, uploaded to the Roboflow project `facadeparsingtext`.

At the same time images are generated with labeled bounding boxes displayed on them, 
they are stored in `ds/grounding_dino_vanilla/output`.

## groundingdino_autodistill.ipynb

```
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
```

ontology is door, front yard, window

`base_model.label` creates a labeled dataset from a sample of 20 images in `ds/grounding_dino_autodistill/base`.
The labeled dataset is stored in the `ds/grounding_dino_autodistill` directory and contains the typical YOLO subdirectories `train/images`, `train/labels` etc. and a .YAML file.

`base_model.predict` calculates bounding boxes, class IDs and confidence scores for the recognized objects.
No large scale experiment was executed to explore this function.

Instead the target model `yolo11n.pt` was trained starting from the labeled dataset. 
This is an object detection model, so it will produce bounding boxes, class IDs and confidence scores.

Training was set up for 400 epochs, however it stopped early at epoch 239 
because no improvements were observed during the preceding 100 epochs.

mAP50-95 was too low

```
                   all          4         30      0.641      0.576      0.616      0.517
                  door          3          4      0.466       0.25      0.275      0.192
            front yard          4          4      0.682       0.75       0.87       0.77
                window          4         22      0.774      0.727      0.703      0.589
```

No large scale experiment was set up using the trained YOLO11 model.


