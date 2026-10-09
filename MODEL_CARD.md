# CurrenSight AI Model Card

## Intended use

CurrenSight AI assists users in identifying the denomination of visible Indian banknotes. It is an accessibility aid, not a financial verification system.

## Current inference stack

- Primary denomination model: ResNet18 with seven output classes.
- Fallback denomination engine: HSV histogram and ORB template matching.
- Localization: heuristic contour detection used only for the visual overlay.
- Authenticity model: two-class ResNet18 research artifact, disabled by default.
- YOLO: offline synthetic-data experiment and not part of the application runtime.

## Classes

₹10, ₹20, ₹50, ₹100, ₹200, ₹500, and ₹2000.

## Data and evaluation limitations

The repository includes a small synthetic localization dataset. It is useful for pipeline tests but provides no evidence of performance on real photographs. The original training images and a leakage-controlled external test set are not included, so the shipped model's real-world accuracy is currently unverified.

Required evaluation conditions include multiple phones, indoor and outdoor lighting, cluttered backgrounds, both note sides, worn or folded notes, partial occlusion, blur, glare, and frames containing no banknote. Results must include per-class precision, recall, F1, confusion matrix, rejection performance, and confidence calibration.

## Risks

- A classifier can confidently assign a denomination to a non-banknote image.
- Contour localization can select rectangular background objects.
- Old, damaged, folded, partially visible, or newly issued notes may be misclassified.
- The authenticity model is not validated for legal or financial use and must remain experimental.
- Confidence is a model score, not a guarantee of correctness.
- A neutral cyan localization box indicates denomination localization only; green/red styling is reserved for explicitly enabled experimental authenticity output.

## Deployment policy

Keep experimental authenticity disabled unless a controlled research evaluation explicitly requires it. User-facing deployments should retain the safety disclaimer and provide a clear unknown/retry state. Any model replacement must be evaluated on an independent photographed-note test set using `evaluation/evaluate_denomination.py`.
