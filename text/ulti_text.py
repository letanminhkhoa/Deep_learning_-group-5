import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def train_and_eval(model, train_loader, test_loader, optimizer, criterion, device, epochs=3, model_type="RNN"):
    print(f"\n{'='*40}")
    print(f"STARTING TRAINING: {model_type}")
    print(f"{'='*40}")
    
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            if model_type == "RNN":
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
            
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Eval]"):
                if model_type == "RNN":
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                else:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1)
                    
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        
        print(f"\nResult Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}\n")
        
    return history