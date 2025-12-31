import os
import argparse
import math
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# --- Model Architectures ---

class CompactBiLSTM(nn.Module):
    def __init__(self, input_dim=960, bottleneck_dim=256, hidden_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(bottleneck_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.projection(x)
        out, _ = self.lstm(x)
        return self.fc(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class InterpretableTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, src):
        src2, attn_weights = self.self_attn(src, src, src, need_weights=True)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src, attn_weights

class TransformerSummarizerV2(nn.Module):
    def __init__(self, input_dim=960, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Dropout(0.3))
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([InterpretableTransformerLayer(d_model, nhead) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, return_attn=False):
        x = self.pos_enc(self.project(x))
        all_attn = []
        for layer in self.layers:
            x, attn = layer(x)
            if return_attn: all_attn.append(attn)
        out = self.head(x)
        return (out, all_attn) if return_attn else out

# --- Helper Functions ---

def load_models(bilstm_path, transformer_path, device):
    print(f"Loading weights from {bilstm_path} and {transformer_path}...")
    
    bilstm = CompactBiLSTM().to(device)
    bilstm.load_state_dict(torch.load(bilstm_path, map_location=device))
    bilstm.eval()
    
    transformer = TransformerSummarizerV2().to(device)
    transformer.load_state_dict(torch.load(transformer_path, map_location=device))
    transformer.eval()
    
    # Feature extractor (Frozen MobileNetV3-Large)
    mobilenet = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
    mobilenet.classifier = nn.Identity()
    mobilenet = mobilenet.to(device)
    mobilenet.eval()
    
    return bilstm, transformer, mobilenet

def extract_features(video_path, model, device, fps=2):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_fps == 0:
        video_fps = 30 # Fallback
        
    step = max(1, int(video_fps / fps))
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    frame_list = []
    features = []
    
    for i in tqdm(range(0, total_frames, step), desc="Extracting features"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(frame_rgb)
        
        img = Image.fromarray(frame_rgb)
        img_t = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model(img_t).cpu().numpy().flatten()
            features.append(feat)
            
    cap.release()
    return np.array(features), frame_list

def select_top_k(scores, budget=0.15):
    n_frames = len(scores)
    k = max(1, int(n_frames * budget))
    top_indices = np.argsort(scores)[-k:]
    mask = np.zeros(n_frames)
    mask[top_indices] = 1
    return mask, top_indices

def run_demo(input_video, output_dir):
    device = torch.device("cpu") # Enforcement of CPU-only inference as per plan
    print(f"Running on {device}")
    
    # Paths (relative to root)
    BILSTM_WEIGHTS = "models/pretrained/tvsum_bilstm_v1.pth"
    TRANSFORMER_WEIGHTS = "models/pretrained/tvsum_transformer_v1.pth"
    
    # Create video-specific output directory
    video_name = Path(input_video).stem
    video_output_path = Path(output_dir) / video_name
    keyframe_dir_bi = video_output_path / "keyframes" / "bilstm"
    keyframe_dir_tr = video_output_path / "keyframes" / "transformer"
    keyframe_dir_bi.mkdir(parents=True, exist_ok=True)
    keyframe_dir_tr.mkdir(parents=True, exist_ok=True)
    
    # Load Models
    bilstm, transformer, mobilenet = load_models(BILSTM_WEIGHTS, TRANSFORMER_WEIGHTS, device)
    
    # Extract Features
    features, frames = extract_features(input_video, mobilenet, device)
    
    # Inference
    feat_t = torch.FloatTensor(features).unsqueeze(0).to(device)
    with torch.no_grad():
        p_bi = torch.clamp(bilstm(feat_t), 0, 1).cpu().squeeze().numpy()
        p_tr, attn = transformer(feat_t, return_attn=True)
        p_tr = torch.clamp(p_tr, 0, 1).cpu().squeeze().numpy()
        
    # Selection
    mask_bi, top_idx_bi = select_top_k(p_bi)
    mask_tr, top_idx_tr = select_top_k(p_tr)
    
    # Visualization Curve
    plt.figure(figsize=(15, 8))
    plt.plot(p_bi, color='blue', label='BiLSTM (Sequential)', alpha=0.8)
    plt.plot(p_tr, color='red', label='Transformer (Global)', alpha=0.8)
    plt.title(f"Importance Predictions for {Path(input_video).name}")
    plt.xlabel("Sampled Frame Index")
    plt.ylabel("Importance Score")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = video_output_path / "demo_plot.png"
    plt.savefig(plot_path)
    print(f"Saved importance plot to: {plot_path}")
    
    # Save Keyframes
    print("Saving keyframes...")
    for idx in top_idx_bi:
        img = Image.fromarray(frames[idx])
        img.save(keyframe_dir_bi / f"kf_{idx:04d}.jpg")
        
    for idx in top_idx_tr:
        img = Image.fromarray(frames[idx])
        img.save(keyframe_dir_tr / f"kf_{idx:04d}.jpg")
        
    print(f"Demo complete for {video_name}! Check {video_output_path}/ for results.")
    print("-" * 30)
    print("Summary:")
    print(f"Total sampled frames: {len(frames)}")
    print(f"BiLSTM selected: {len(top_idx_bi)} frames")
    print(f"Transformer selected: {len(top_idx_tr)} frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean E2E Video Summarization Demo")
    parser.add_argument("--input", type=str, required=True, help="Path to input video (.mp4) or directory")
    parser.add_argument("--output", type=str, default="results", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if input_path.is_dir():
        video_files = list(input_path.glob("*.mp4"))
        if not video_files:
            print(f"Error: No .mp4 files found in {args.input}")
        else:
            print(f"Found {len(video_files)} videos in {args.input}")
            for video_file in video_files:
                run_demo(str(video_file), args.output)
    elif input_path.exists():
        run_demo(args.input, args.output)
    else:
        print(f"Error: Path {args.input} not found.")
