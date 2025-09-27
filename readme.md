# Human Emotion Detection (Computer Vision)

This project leverages **ResNet50 deep learning architecture** to automatically classify facial images into three emotions: **Angry, Happy, and Sad**. The system aims to automate traditional emotion recognition.

The model was trained on a dataset of **9,077 labeled facial images** (6,799 for training and validation, 2,278 for testing). Data augmentation strategies such as **rotation, flipping, zoom, translation, contrast, and brightness adjustments** were applied to improve generalization. Additionally, a **CutMix-inspired augmentation** was used to enhance generalization.

Training used **transfer learning** with ResNet50 pretrained on ImageNet:

1. **Phase 1:** Freeze base layers and train only the classifier head for 5 epochs.
2. **Phase 2:** Fine-tune the last 75 layers of the base model for 20 epochs with **AdamW optimizer and cosine decay learning rate**.

The model achieved **88.7% test accuracy**, with a strong balance between **precision (88–92%)** and **recall (82–92%)** across all three classes. Confusion matrix analysis shows reliable class separation, especially for Happy and Sad emotions.

**Dataset source:** [Human Emotions Dataset(HES)](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes)

**Project documentation:** [View full project documentation and demo](https://adittomahmood.vercel.app/project-docs/2)

**Live app:** [Try the human emotion detection system](https://human-emotion-detection.streamlit.app/)

**Connect:** [LinkedIn](https://linkedin.com/in/adittomahmood)
