# Traffic Sign Recognition System

This project implements a **two-stage Traffic Sign Recognition (TSR) pipeline**:

1. **Detection/Segmentation** of traffic signs in input images (complex backgrounds).
2. **Classification** of the detected signs into their correct categories.

We compare two classification models:

- **Custom CNN**: Designed and trained from scratch.
- **EfficientNet**: A pretrained state-of-the-art model fine-tuned on the dataset.

The project aims to benchmark traditional CNNs versus modern scalable architectures (EfficientNet) for traffic sign classification.

---

## Dataset

We use the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset:

- \~50,000 images of traffic signs
- 43 classes
- Variable lighting, background, and resolution conditions

Dataset link: [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html)

---

## Models

### 1. Custom CNN (Baseline)

- **Architecture**:
  - Convolutional layers with ReLU + MaxPooling
  - Dropout layers for regularization
  - Fully connected dense layers
- **Training**:
  - Optimizer: Adam
  - Loss: Categorical Crossentropy
  - Accuracy: \~95% (expected)

This model demonstrates how a simple CNN can perform well on TSR tasks.

### 2. EfficientNet (Pretrained)

- **Architecture**:
  - EfficientNet-B0 (transfer learning)
  - Scaled with compound scaling of depth, width, resolution
- **Training**:
  - Fine-tuned last layers on GTSRB dataset
  - Optimizer: AdamW
  - Accuracy: \~98%+ (expected)

EfficientNet provides higher accuracy and efficiency, showing benefits of modern architectures.

---

## Pipeline Overview

1. **Input**: Traffic scene image
2. **Segmentation/Detection**: Extract traffic signs (bounding box / segmentation mask)
3. **Classification**: Use Custom CNN and EfficientNet for label prediction
4. **Comparison**: Evaluate both models on accuracy, inference time, and robustness

---

## Results (Expected)

| Model        | Accuracy | Training Time | Inference Speed |
| ------------ | -------- | ------------- | --------------- |
| Custom CNN   | \~95%    | Fast          | Very fast       |
| EfficientNet | \~98%+   | Moderate      | Fast            |

---

## Research Contributions

- Comparison of baseline CNN vs advanced EfficientNet on traffic sign recognition.
- Demonstrates importance of modern scaling approaches in real-world AI tasks.
- Provides an end-to-end TSR pipeline for research and industry applications.

---

## Future Work

- Integrate YOLOv8 / DeepLabV3+ for segmentation (detection stage).
- Test robustness against adversarial attacks and occlusions.
- Optimize EfficientNet for mobile deployment (TensorRT, TFLite).

---

## How to Run

```bash
# Clone repository
git clone <repo-link>
cd traffic-sign-recognition

# Install dependencies
pip install -r requirements.txt

# Train Custom CNN
python train_cnn.py

# Train EfficientNet
python train_efficientnet.py

# Evaluate models
python evaluate.py
```

---

## References

- [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html)
- Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking model scaling for convolutional neural networks." ICML 2019.
- Krizhevsky et al. "ImageNet classification with deep convolutional neural networks." NIPS 2012.

---

## Author

Kunnu

