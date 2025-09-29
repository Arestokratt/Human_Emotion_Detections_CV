# Human Emotion Detection (Computer Vision)

- **Dataset**:

  - Total: **26,404 labeled images**
  - Train/Validation: **19,278 images (80/20 split)**
  - Test: **7,126 images**
  - Source: [Human Emotions Dataset (HES)](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes)

- **Data Augmentation**:
  Random rotation, flipping, zoom, translation, contrast, and brightness adjustments.
  Additionally,**CutMix-inspired augmentation** was applied.

- **Training Strategy**:

  1. **Phase 1** – Freeze base ResNet50 layers, train classifier head for 10 epochs.
  2. **Phase 2** – Fine-tune last 75 layers of ResNet50 for 25 epochs.

  - Optimizer: **AdamW**
  - Learning rate: **Cosine decay + ReduceLROnPlateau**
  - Regularization: **L2 weight decay**

## Results

- **Test Accuracy**: **91.4%**

## Demo

- **Live App**: [Try it here](https://human-emotion-detection-cv.streamlit.app/)
- **Full Project Docs**: [Read here](https://adittomahmood.vercel.app/project-docs/5)

<div align="center">
  <h2>🌐 LET'S CONNECT & COLLABORATE</h2>
  <a href="https://linkedin.com/in/adittomahmood">
    <img src="https://custom-icon-badges.demolab.com/badge/_LinkedIn-0077B5?style=for-the-badge&logoColor=white&logo=linkedin" />
  </a>
  <a href="https://github.com/adittomahmood">
    <img src="https://custom-icon-badges.demolab.com/badge/_Follow_Me-181717?style=for-the-badge&logoColor=white&logo=github" />
  </a>
  <br/><br/>
  <p style="font-size: 12px; color: #8B949E;">
    © 2025 Tasneem Bin Mahmood • Machine Learning & Computer Vision Engineer
  </p>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/Trilokia/Trilokia/379277808c61ef204768a61bbc5d25bc7798ccf1/bottom_header.svg" width="100%" />
</div>
