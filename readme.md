<html lang="en">
<head>
  <meta charset="utf-8">
</head>
<body>
  <h1>Human Emotion Detection (Computer Vision)</h1>
  <h2>Dataset</h2>
  <ul>
    <li>Total: <strong>26,404 labeled images</strong></li>
    <li>Train/Validation: <strong>19,278 images (80/20 split)</strong></li>
    <li>Test: <strong>7,126 images</strong></li>
    <li>Source: <a href="https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes">Human Emotions Dataset (HES)</a></li>
  </ul>
  <h2>Data Augmentation</h2>
  <p>
    Random rotation, flipping, zoom, translation, contrast, and brightness adjustments.
    Additionally, <strong>CutMix-inspired augmentation</strong> was applied.
  </p>
  <h2>Training Strategy</h2>
  <ol>
    <li><strong>Phase 1</strong> – Freeze base ResNet50 layers, train classifier head for 10 epochs.</li>
    <li><strong>Phase 2</strong> – Fine-tune last 75 layers of ResNet50 for 25 epochs.</li>
  </ol>
  <ul>
    <li>Optimizer: <strong>AdamW</strong></li>
    <li>Learning rate: <strong>Cosine decay + ReduceLROnPlateau</strong></li>
    <li>Regularization: <strong>L2 weight decay</strong></li>
  </ul>
  <h2>Results</h2>
  <ul>
    <li><strong>Test Accuracy</strong>: 91.4%</li>
  </ul>
  <h2>Demo</h2>
  <ul>
    <li><strong>Live App</strong>: <a href="https://human-emotion-detection-cv.streamlit.app/">Try it here</a></li>
    <li><strong>Full Project Docs</strong>: <a href="https://adittomahmood.vercel.app/human-emotion-detection-cv">Read here</a></li>
  </ul>
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
</body>
</html>
