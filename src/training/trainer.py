"""Training loop for rPPG model."""

import torch
import torch.optim as optim
from pathlib import Path
from src.models.losses import MultiTaskLoss
from src.training.early_stopping import EarlyStopping
from src.training.logger import TrainingLogger


class Trainer:
    """Training loop for rPPG model with early stopping and checkpointing."""

    def __init__(self, model, device, checkpoint_dir: Path, log_dir: Path):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = TrainingLogger(log_dir)
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader, optimizer, loss_fn) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch, (tuple, list)) and len(batch) == 4:
                frames, hr_target, rr_target, spo2_target = batch
            else:
                continue

            frames = frames.to(self.device)
            hr_target = hr_target.to(self.device)
            rr_target = rr_target.to(self.device)
            spo2_target = spo2_target.to(self.device)

            # Forward pass
            outputs = self.model(frames)
            targets = (hr_target, rr_target, spo2_target)
            loss = loss_fn(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def validate(self, val_loader, loss_fn) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 4:
                    frames, hr_target, rr_target, spo2_target = batch
                else:
                    continue

                frames = frames.to(self.device)
                hr_target = hr_target.to(self.device)
                rr_target = rr_target.to(self.device)
                spo2_target = spo2_target.to(self.device)

                outputs = self.model(frames)
                targets = (hr_target, rr_target, spo2_target)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(self, train_loader, val_loader, epochs: int = 150, lr: float = 1e-4):
        """Full training loop."""
        loss_fn = MultiTaskLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        early_stopping = EarlyStopping(patience=15)

        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, loss_fn)
            val_loss = self.validate(val_loader, loss_fn)
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            self.logger.log_epoch(epoch, train_loss, val_loss, current_lr)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")

            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch)
                print(f"  → Checkpoint saved (val_loss: {val_loss:.4f})")

            # Early stopping check
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        print(f"\nTraining complete. Best validation loss: {self.best_val_loss:.4f}")
        self.logger.save()
        self.save_final_model()

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")

    def save_final_model(self):
        """Save final trained model."""
        torch.save(self.model.state_dict(), self.checkpoint_dir / "final_model.pt")
        print(f"Final model saved to {self.checkpoint_dir / 'final_model.pt'}")
