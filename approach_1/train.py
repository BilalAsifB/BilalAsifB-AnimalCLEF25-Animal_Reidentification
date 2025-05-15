import torch
import torch.optim as optim
from .loss import contrastive_loss

def train(model, train_loader, val_loader, config):
    """Train the Siamese Network with early stopping (CPU-only)."""
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    if len(val_loader) == 0:
        print("Warning: Validation loader is empty. Training without validation.")
        num_epochs = min(config['num_epochs'], 10)
    else:
        num_epochs = config['num_epochs']
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_idx, (img1, img2, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = contrastive_loss(output1, output2, label, margin=config['margin'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        
        if len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img1, img2, label in val_loader:
                    output1, output2 = model(img1, img2)
                    loss = contrastive_loss(output1, output2, label, margin=config['margin'])
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), config['model_path'])
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: N/A')
            torch.save(model.state_dict(), config['model_path'])
        
        model.train()