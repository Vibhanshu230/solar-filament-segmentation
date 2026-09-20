# Solar Filament Segmentation

Kaggle: [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)

## 1. What this competition is about

The task is to find **solar filaments** in full-disk H-alpha telescope images and draw a separate outline for each one.

A filament is a dark, thin structure on the Sun. The score is not only “did you paint the right pixels?” It also checks whether each predicted blob matches one real filament. Extra pieces and missed filaments both hurt the score (Panoptic Quality).

Data comes from MAGFiLO / GONG. Images are 2048×2048 grayscale. Labels are polygons for each filament.

## 2. Two approaches I tried

### U-Net (main submission)

Notebook: `unet_pipeline.ipynb`

1. Clean the image (solar disk mask, limb-darkening correction, contrast).
2. Train a ResNet-34 U-Net on 512×512 tiles to mark filament pixels.
3. Turn the probability map into objects with connected components (not watershed).
4. Write each object as RLE in `submission.csv`.

**Results**

- Validation: PQ **0.38**, Dice **0.64**
- Kaggle public score: **0.33** (got 296 rank at the time)

This is the model I submitted.

### YOLO segmentation

Notebook: `yolo_segmentation_code/yolo-seg.ipynb`

YOLO predicts boxes and masks in one pass, so it does not need a separate “cut the blobs” step.

- `yolo11n-seg` at image size 640: val PQ **0.21**
- Same model at image size 1024: val PQ **0.32**

Better than 640, still below the U-Net. I kept the U-Net CSV.

## 3. Challenges

- **Watershed over-cut.** It split long filaments into many small pieces. Validation PQ was about **0.08**. Switching to connected components and dropping tiny blobs (`min_area=500`) raised PQ to **0.38**.
- **U-Net vs contest metric.** Tile loss looked good (~0.17) while instance score was still poor until post-processing was fixed.
- **YOLO resolution.** At 640, thin filaments were too small. 1024 helped (0.21 → 0.32) but did not beat the U-Net.

## 4. What I would try next

**Mask R-CNN.** Unlike the U-Net, it is trained to output objects directly: a box, a mask, and a score for each filament. That may reduce the need for hand-tuned cutting (`threshold`, `min_area`).

Other follow-ups: a larger YOLO (`yolo11s-seg` at 1024), or a light merge step for leftover splits.

## How to run

1. Download the competition data with KaggleHub into `MAGFiLO_1.0_Kaggle_2026/`.
2. Install packages from `requirements.txt`.
3. Run `unet_pipeline.ipynb` for the submitted pipeline.
4. Run `yolo_segmentation_code/yolo-seg.ipynb` for the YOLO experiment.
