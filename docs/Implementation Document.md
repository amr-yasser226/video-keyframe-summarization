## Deep Learning–Based Keyframe Detection

**Datasets:** TVSum & SumMe  
**Course:** Deep Learning — Fall 2025

---

## 1. Project Overview

### 1.1 Objective

The objective of this project is to design and implement a **deep learning–based keyframe detection system** that automatically identifies the most representative and informative frames from video content.

Two distinct temporal modeling techniques are implemented and compared:

1. **Bidirectional LSTM–based importance regression**
    
2. **Transformer (self-attention)–based summarization**
    

The system is evaluated quantitatively on benchmark datasets and qualitatively on diverse real-world videos.

---

### 1.2 Definition of Keyframes

In this project, **keyframes** are defined as a small, ordered subset of frames that collectively:

- **Maximize representativeness** of the video’s main content and events
    
- **Maximize diversity** while avoiding near-duplicate frames
    
- **Minimize redundancy** across time
    
- **Preserve temporal coverage**, maintaining the storyline structure
    

**System Input:**

- A colored video clip (30–60 seconds for demo; longer videos for benchmarks)
    

**System Output:**

- A fixed number of keyframes per video (default **K = 10**, configurable), each with:
    
    - Frame index
        
    - Timestamp (seconds)
        
    - Image file (JPG/PNG)
        

---

### 1.3 Scope

The project covers the **complete video summarization pipeline**:

- Dataset preparation and annotation alignment
    
- Frame extraction and preprocessing
    
- Deep feature extraction using pretrained CNNs
    
- Temporal importance modeling using two architectures
    
- Keyframe selection and visualization
    
- Quantitative and qualitative evaluation
    

All development is conducted using **cloud-based GPU environments** (Google Colab and Kaggle), with Google Drive used for persistent storage.

---

## 2. Problem Analysis

### 2.1 Core Challenges

**Challenge 1 — Defining “Importance”**  
Frame importance is subjective and varies by:

- Video domain (sports, news, cooking)
    
- Viewer intent
    
- Temporal context
    

**Solution:**  
Use **human-annotated importance scores** from TVSum as supervision, capturing average human perception.

---

**Challenge 2 — Temporal Dependencies**  
Frame importance depends on surrounding context (e.g., buildup → climax).

**Solution:**  
Employ **sequence models (BiLSTM and Transformer)** that explicitly model temporal relationships.

---

**Challenge 3 — Domain Generalization**  
Models trained on one dataset may fail on unseen domains.

**Solution:**  
Train on **TVSum**, evaluate generalization on **SumMe**, and test on diverse demo videos.

---

**Challenge 4 — Computational Constraints**  
End-to-end video models are expensive.

**Solution:**  
Adopt a **two-stage pipeline**:

1. Frozen CNN feature extraction (run once)
    
2. Lightweight temporal modeling (trainable)
    

This reduces training time from hours to minutes.

---

## 3. Datasets

### 3.1 Training & Validation Dataset — TVSum

- **50 videos**, 10 categories
    
- Duration: **1–10 minutes**
    
- **20 annotators per video**
    
- Frame-level importance scores (1–5 scale)
    

**Processing:**

- Scores averaged across annotators
    
- Normalized to [0, 1]
    
- Aligned to resampled frames (2 FPS)
    

**Split Strategy:**

- 40 videos for training
    
- 10 videos for validation
    
- Split by category to test cross-category generalization
    

---

### 3.2 Test Dataset — SumMe

- **25 videos**, mixed categories
    
- Duration: **1–6 minutes**
    
- Ground truth provided as **user-selected video segments**
    

**Usage:**

- Used exclusively for final evaluation
    
- No training or hyperparameter tuning
    
- Measures cross-dataset generalization
    

---

### 3.3 Demo Videos

Four short clips (30–60s) used for qualitative evaluation:

|Domain|Content|
|---|---|
|Sports|Football goal highlight|
|News|News broadcast segment|
|Driving|Dashcam footage|
|Cooking|Recipe steps|

Only **Creative Commons or public-domain sources** are used. The repository stores extracted frames, not raw copyrighted videos.

---

## 4. System Overview

### 4.1 Unified Pipeline

```
Input Video
   ↓
Frame Extraction (2 FPS, 224×224)
   ↓
CNN Feature Extraction (frozen)
   ↓
Temporal Importance Model
   ↓
Frame Importance Scores
   ↓
Keyframe Selection (Top-K + temporal suppression)
   ↓
Keyframes + timestamps + visualization
```

Both models share **identical preprocessing, features, and post-processing** to ensure fair comparison.

---

## 5. Data Processing Pipeline

### 5.1 Video Standardization

For all videos:

- Decode video
    
- Sample at **2 FPS**
    
- Resize to **224×224**
    
- Apply ImageNet normalization
    
- Store frames and metadata (timestamps, FPS, frame count)
    

**Why 2 FPS?**

- Standard in video summarization literature
    
- Reduces computation by ~12–15×
    
- Preserves temporal structure
    

---

### 5.2 Annotation Alignment

TVSum annotations (original FPS) are downsampled to match extracted frames.  
Scores are averaged across annotators and normalized to [0, 1].

---

## 6. Feature Extraction

### 6.1 Rationale

Training end-to-end video models is impractical under resource constraints. Instead:

- Extract deep visual features **once**
    
- Train temporal models on compact representations
    

---

### 6.2 Feature Extractors

**Primary:** GoogLeNet (Inception v1)

- Feature dimension: 1024
    
- Widely used in video summarization literature (VASNet, SUM-GAN)
    

**Alternative:** ResNet-50 (2048-D)

Features are stored as `.npy` files of shape **T × D**.

---

## 7. Temporal Models

### 7.1 Model A — BiLSTM Importance Regressor

**Key Idea:**  
Each frame’s importance depends on **past and future context**.

**Architecture:**

- Input projection
    
- 2-layer BiLSTM
    
- MLP head with sigmoid output
    

**Training Objective:**

- Supervised regression using **MSE or Huber loss**
    

**Strengths:**

- Strong baseline
    
- Stable training
    
- Good at modeling local temporal continuity
    

---

### 7.2 Model B — Transformer Summarizer

**Key Idea:**  
Use **self-attention** to model global relationships across all frames.

**Architecture:**

- Linear embedding
    
- Positional encoding
    
- Transformer encoder blocks
    
- MLP output head
    

**Strengths:**

- Captures long-range dependencies
    
- Reduces redundancy
    
- Attention weights offer interpretability
    

---

### 7.3 Model Comparison

|Aspect|BiLSTM|Transformer|
|---|---|---|
|Temporal modeling|Sequential|Global|
|Long-range context|Limited|Strong|
|Training stability|High|Needs tuning|
|Interpretability|Low|Attention maps|
|Complexity|O(T)|O(T²)|

---

## 8. Training Procedure

- Loss: Mean Squared Error (masked for padding)
    
- Optimizer: Adam
    
- Gradient clipping for stability
    
- Learning rate scheduling (cosine annealing)
    
- Best model selected via validation loss
    

---

## 9. Keyframe Selection

### 9.1 Algorithm — Top-K with Temporal Suppression

1. Rank frames by predicted importance
    
2. Select highest-scoring frames
    
3. Enforce minimum temporal distance (Δ seconds)
    
4. Stop when **K** frames are selected
    
5. Sort selected frames by time
    

**Default Parameters:**

- K = 10
    
- Δ = 1–2 seconds
    

This prevents near-duplicate selections and improves readability.

---

## 10. Evaluation

### 10.1 Quantitative Metrics

- **F1 score** (SumMe)
    
- **Rank correlation** (TVSum)
    
- **Diversity** (pairwise embedding distance)
    
- **Coverage** (representation of full video)
    

---

### 10.2 Qualitative Analysis

- Side-by-side keyframe grids
    
- Timeline visualization of selected frames
    
- Failure case discussion (missed events, redundancy)
    

---

## 11. Experiments

1. BiLSTM vs Transformer on TVSum validation
    
2. Generalization evaluation on SumMe
    
3. Sensitivity analysis:
    
    - K ∈ {5, 10, 15}
        
    - FPS ∈ {1, 2, 5}
        
    - Temporal suppression Δ
        
4. Demo video comparisons
    

---