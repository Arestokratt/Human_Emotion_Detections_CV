# Human Emotion Detection (Computer Vision)

* **Dataset**:

  * Total: **26,404 labeled images**
  * Train/Validation: **19,278 images (80/20 split)**
  * Test: **7,126 images**
  * Source: [Human Emotions Dataset (HES)](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes)

* **Data Augmentation**:
  Random rotation, flipping, zoom, translation, contrast, and brightness adjustments.
  Additionally,**CutMix-inspired augmentation** was applied.

* **Training Strategy**:

  1. **Phase 1** – Freeze base ResNet50 layers, train classifier head for 10 epochs.
  2. **Phase 2** – Fine-tune last 75 layers of ResNet50 for 25 epochs.

  * Optimizer: **AdamW**
  * Learning rate: **Cosine decay + ReduceLROnPlateau**
  * Regularization: **L2 weight decay**

## Results

* **Test Accuracy**: **91.4%**



## Demo 

* **Live App**: [Try it here](https://human-emotion-detection-cv.streamlit.app/)
* **Full Project Docs**: [Read here](https://adittomahmood.vercel.app/project-docs/5)

## Connect

* [LinkedIn](https://linkedin.com/in/adittomahmood)
