"""
ml_engine/train.py
===================
Training script for the CNN-LSTM Elastography model.
Includes support for:
  - CUDA / GPU acceleration
  - Gradient Accumulation
  - Checkpointing (saving and resuming)
  - TensorBoard logging

Usage:
  python train.py --epochs 50 --batch_size 8 --accumulate_steps 4 --lr 1e-4
"""

import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from architecture import CNNLSTMElastography

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM Elastography Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--accumulate_steps", type=int, default=4, help="Number of steps for gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training")
    parser.add_argument("--save_dir", type=str, default="weights", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for TensorBoard logs")
    return parser.parse_args()

def dummy_dataloader(batch_size, num_samples=100):
    """
    Creates a dummy dataloader for demonstration.
    Replace this with your actual dataset loading logic (e.g., from .mat files).
    Expected shape for model input: (B, T, 1, H, W)
    """
    # Assuming T=8, H=64, W=64 based on inference.py
    T, H, W = 8, 64, 64
    x = torch.randn(num_samples, T, 1, H, W)
    y = torch.randint(0, 3, (num_samples,)) # 3 classes: Benign, Precancerous, Malignant
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    args = get_args()

    # 1. CUDA Support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # 2. Initialize Model, Loss, Optimizer
    model = CNNLSTMElastography(
        cnn_channels=[1, 16, 32, 64, 128],
        lstm_hidden=256,
        lstm_layers=2,
        n_classes=3,
        patch_size=64,
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0

    # 3. Checkpointing - Resume if specified
    if args.resume and os.path.isfile(args.resume):
        print(f"[Train] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"[Train] Resumed at epoch {start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 4. Train Logging - TensorBoard
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"run_{timestamp}"))

    # Replace with real DataLoader
    train_loader = dummy_dataloader(args.batch_size)
    
    print(f"[Train] Starting training for {args.epochs} epochs with accumulation steps = {args.accumulate_steps}")

    model.train()
    
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        
        optimizer.zero_grad() # Zero gradients at start of epoch
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits, displacement_t = model(inputs)
            
            # Loss computation
            loss = criterion(logits, targets)
            
            # 5. Gradient Accumulation
            # Normalize the loss to account for accumulation
            loss = loss / args.accumulate_steps
            
            # Backward pass
            loss.backward()

            # Step optimizer after accumulating gradients
            if (i + 1) % args.accumulate_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
            epoch_loss += loss.item() * args.accumulate_steps

            # Log batch loss
            writer.add_scalar('Loss/train_step', loss.item() * args.accumulate_steps, global_step)

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)

        # Save Checkpoint at the end of each epoch
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        
        # Keep a 'latest' or 'final' model copy updated
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.save_dir, "cnn_lstm_final.pth"))

    writer.close()
    print("[Train] Training complete.")

if __name__ == "__main__":
    main()
