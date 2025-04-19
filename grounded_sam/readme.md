## groundedsam_autodistill.ipynb and groundedsam_autodistill_colab.ipynb

```
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
```

ontology is door, front yard, window

Without GPU, `base_model.label` took 200 seconds for one image. With a T4 GPU a sample of 20 images was processed in 90 seconds.

`base_model.predict` calculates bounding boxes, masks, class IDs and confidence scores for the recognized objects 
in each of the 181 images in the `ds/groundedsam_autodistill/full` directory.
These results were stored in .hdf5 files that were subsequently downloaded to a laptop for further analysis.

In the output directory `ds/groundedsam_autodistill/full_output` two output images are stored for each input image:
- annotated image (original image with the segmentation masks superposed on it)
- black image with only the segmentation masks for greater clarity


