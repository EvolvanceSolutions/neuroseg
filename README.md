# neuroseg
Brain tumor AI detector
# NeuroSeg â€” Brain Tumor Segmentation from MRI

U-Net based semantic segmentation for lower-grade glioma (LGG) detection in brain MRI.
Built over 10 days with systematic ablation â€” every improvement measured independently.

![Results](results.png)

---

## Results

| # | Setup | Tumor Dice | Complete Miss | Empty-scan Dice |
|---|---|---|---|---|
| 1 | Baseline U-Net (from scratch) | 0.3598 | 44.4% | 0.9752 |
| 2 | + Weighted Sampler | 0.4564 | 25.8% | 0.2650 |
| 3 | + Augmentation | 0.5042 | 24.7% | 0.5280 |
| 4 | + ResNet34 encoder (ImageNet) | 0.7878 | 3.7% | 0.7308 |
| 5 | **+ Post-processing (FINAL)** | **0.7832** | **4.4%** | **0.7847** |

**Headline:** Complete miss rate reduced from **44.4% to 4.4%** â€” a 10x
improvement in the metric that matters clinically. Dice improved +118%.

---

## The three bugs that mattered

**1. The metric was lying.**
Initial Dice of 0.7438 looked fine. But 65% of validation slices contain no
tumor, and an empty prediction on an empty mask scores 1.0. The model was being
rewarded for silence. Measuring tumor-containing slices separately revealed the
real score: **0.3598**.

**2. The model learned a shortcut, not the task.**
Visual error analysis showed predictions only fired where the FLAIR channel was
bright. The model had learned to detect a colour, not a tumour. Augmentation
(flip / rotate / intensity) broke that shortcut.

**3. Counting the wrong thing.**
Post-processing removed 94% of false-positive *components* but improved the score
by only 0.07 â€” because Dice is computed per *scan*, and one surviving large blob
fails the whole scan regardless of how many small ones were cleaned.

---

## Method

| | |
|---|---|
| **Dataset** | LGG Segmentation (Buda et al.) â€” 110 patients, 3,929 slices |
| **Split** | Patient-level, 88 train / 22 val (never slice-level â€” adjacent slices are near-identical) |
| **Input** | 3 channels = pre-contrast, FLAIR, post-contrast (not RGB) |
| **Architecture** | U-Net, ResNet34 encoder, ImageNet pretrained |
| **Loss** | 0.5 Ã— Dice + 0.5 Ã— BCE |
| **Sampling** | Weighted â€” tumour slices oversampled 1.92x to ~50/50 |
| **Augmentation** | H-flip, rotation Â±15Â°, zoom Â±10%, intensity Â±15%. **No colour jitter** â€” the channels are MRI sequences, not colours |
| **Optimizer** | Adam, lr 1e-4 (10x lower for pretrained encoder), AMP |
| **Post-processing** | threshold 0.5, remove connected components < 100 px |

---

## Evaluation protocol

Tumour-containing and empty slices are reported **separately**. A single averaged
Dice is misleading on this dataset â€” empty slices score ~1.0 for free and inflate
the headline number by roughly 0.38.

The primary clinical metric is **complete miss rate**: the fraction of
tumour-containing slices where the model outputs nothing. A false positive costs
a radiologist 30 seconds; a false negative is a missed tumour.

---

## Known limitations

- **Dice penalises near-misses as hard as complete misses.** Two validation cases
  score 0.000 despite the model flagging roughly the right region. Lesion-wise
  metrics (as used in recent BraTS challenges) would capture this better.
- **~22% of clean scans still receive a false positive.** Below the level where a
  radiologist would trust unsupervised output.
- **Single institution, 2D, LGG only.** No generalisation claim to GBM,
  paediatric tumours, other scanners, or 3D volumetric context.
- **Threshold and component-size cutoff tuned on the validation set.** A held-out
  test split would give an unbiased estimate.

---

## Reproduce

```python
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load("unet_C_resnet34.pth"))
model.eval()

prob = torch.sigmoid(model(x))
pred = remove_small_components(prob > 0.5, min_size=100)
```

Seed 42 throughout. Verified reproducible: reloading the checkpoint across
sessions reproduces 0.7878 / 3.7% / 0.7308 exactly.

---

## RESEARCH USE ONLY â€” NOT A MEDICAL DEVICE

Not validated for clinical use. Not cleared by any regulatory body. Must not be
used for diagnosis or treatment decisions.

## Citation

Buda M., Saha A., Mazurowski M.A. (2019). *Association of genomic subtypes of
lower-grade gliomas with shape features automatically extracted by a deep
learning algorithm.* Computers in Biology and Medicine. Dataset: CC BY-NC-SA 4.0
