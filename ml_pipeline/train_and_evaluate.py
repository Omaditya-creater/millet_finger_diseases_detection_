import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import FingerMilletDataset, get_transforms, CLASSES
from model import build_model
from gradcam import GradCAMGenerator
from genai_advisor import GenAIAgronomicAdvisor

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += images.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    total = len(all_labels)
    eval_loss = running_loss / total
    eval_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return eval_loss, eval_acc, np.array(all_preds), np.array(all_labels)

def main():
    print("=" * 60)
    print(" Finger Millet Disease Detection: Hybrid Model & GenAI Pipeline ")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Utilizing Compute Device: {device}")
    
    # 1. Prepare Datasets & DataLoaders
    train_tf, val_tf = get_transforms()
    train_dataset = FingerMilletDataset(transform=train_tf, num_synthetic_per_class=10)
    val_dataset = FingerMilletDataset(transform=val_tf, num_synthetic_per_class=5)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"[*] Train set size: {len(train_dataset)} | Validation set size: {len(val_dataset)}")
    
    # 2. Build Hybrid Model
    model = build_model(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 3. Train Model
    epochs = 10
    print(f"\n[*] Starting Hybrid Model Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
        
    # 4. Final Evaluation Metrics
    val_loss, val_acc, preds, targets = evaluate(model, val_loader, criterion, device)
    print("\n" + "=" * 60)
    print(" PERFORMANCE EVALUATION METRICS ")
    print("=" * 60)
    print(f"Final Validation Accuracy: {val_acc * 100:.2f}%")
    
    # Save Checkpoint
    save_path = os.path.join(os.path.dirname(__file__), "hybrid_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[*] Saved Trained Model Weights to: {save_path}")
    
    # 5. Test Grad-CAM & GenAI Integration
    print("\n[*] Testing Explainable AI (Grad-CAM) & GenAI Advisory Module...")
    gradcam = GradCAMGenerator(model)
    sample_img, sample_lbl = val_dataset[0]
    sample_tensor = sample_img.unsqueeze(0).to(device)
    
    cam_res = gradcam.generate_heatmap(sample_tensor)
    pred_cls_name = CLASSES[cam_res['target_class']]
    
    advisor = GenAIAgronomicAdvisor()
    advisory = advisor.generate_advisory(
        disease_class=pred_cls_name,
        confidence=cam_res['confidence'],
        lsi_percentage=cam_res['lsi_percentage']
    )
    
    print("\n" + "=" * 60)
    print(" GENAI AGRONOMIC ADVISORY REPORT ")
    print("=" * 60)
    print(advisory)
    print("=" * 60)
    print("[*] Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
