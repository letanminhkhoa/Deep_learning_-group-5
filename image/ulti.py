import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate(model,test_loader,criterion,device):
    model.eval()
    correct_pred, total_pred, current_loss = 0,0,0
    wrong_examples = []
    y_true = []
    y_pred = []
    all_probs = []
    with torch.no_grad():
        for image, label, original_img in tqdm(test_loader,desc="Evaluating"):
            image, label = image.to(device),label.to(device)
            output = model(image)
            loss = criterion(output,label)
            current_loss += loss.item()
            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            correct_pred += (predicted == label).sum().item()
            total_pred += label.size(0)
            mismatches = predicted != label
            if mismatches.any():
                wrong_examples.extend([(label[i], predicted[i], original_img[i]) for i in range(len(label)) if mismatches[i]])
            all_probs.append(probs.cpu())
            y_true.extend(label.tolist())
            y_pred.extend(predicted.tolist())
    all_probs = torch.cat(all_probs).numpy()
    accuracy = correct_pred/total_pred
    return current_loss, accuracy, all_probs, wrong_examples, y_true, y_pred

def train(model,train_loader,criterion,optimizer,device,epochs,val_loader=None):
    model.train()
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    for epoch in range(epochs):
        correct_pred, total_pred, current_train_loss = 0,0,0
        for image, label, _ in tqdm(train_loader,desc=f"Training {epoch+1}/{epochs}"):
            image, label = image.to(device),label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output,label)
            current_train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_pred += (predicted == label).sum().item()
            total_pred += label.size(0)
            loss.backward()
            optimizer.step()
        train_loss.append(current_train_loss)
        train_accuracy.append(correct_pred/total_pred)
        if val_loader is not None:
            current_val_loss, current_val_acc, _, _, _, _ = evaluate(model,val_loader,criterion,device)
            val_loss.append(current_val_loss)
            val_accuracy.append(current_val_acc)
            print(f"""epoch: [{epoch+1}/{epochs}], train loss: {current_train_loss:4f}, train acc: {train_accuracy[-1]:4f}, val loss: {current_val_loss:4f}, val acc: {current_val_acc:4f}""")
        else:
            print(f"""epoch: [{epoch+1}/{epochs}], train loss: {current_train_loss:4f}, train acc: {train_accuracy[-1]:4f}""")
    return train_loss, train_accuracy, val_loss, val_accuracy

def ensemble_predict(CNN_model, vit_model, test_loader, CNN_weights = 0.5, vit_weights = 0.5,device ="cpu"):
    CNN_model.eval()
    vit_model.eval()
    y_true = []
    CNN_y_pred = []
    ViT_y_pred = []
    en_y_pred = []
    with torch.no_grad():
        for images, label, _ in tqdm(test_loader, desc="Ensemble Predicting"):
            images, label = images.to(device),label.to(device)
            cnn_probs = F.softmax(CNN_model(images), dim=1)
            vit_probs = F.softmax(vit_model(images), dim=1)

            final_probs = CNN_weights * cnn_probs + vit_weights * vit_probs
            y_true.extend(label.tolist())
            CNN_y_pred.extend(torch.argmax(cnn_probs, dim=1).tolist())
            ViT_y_pred.extend(torch.argmax(vit_probs, dim=1).tolist())
            en_y_pred.extend(torch.argmax(final_probs, dim=1).tolist())
    return CNN_y_pred, ViT_y_pred, en_y_pred, y_true