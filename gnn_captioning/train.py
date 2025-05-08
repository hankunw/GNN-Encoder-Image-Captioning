
import torch
from tqdm import tqdm
from config import device, EPOCHS
from evaluate import evaluateBLEU




def train_model(model, name,train_loader, val_loader, loss_func, optimizer, epochs=EPOCHS, use_scheduler=True):
    model = model.to(device)
    
    # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    
    best_bleu = 0
    train_losses = []
    val_losses = []
    val_bleu = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',    
        factor=0.2,    
        patience=2,    
        verbose=True   
    )
    
    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch in progress_bar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            pad_masks = batch['attention_mask'].to(device)
            
            # forward
            outputs = model(images, input_ids[:, :-1],pad_masks[:,:-1])  # 使用teacher forcing
            
            # loss
            logits = outputs.view(-1, outputs.size(-1))
            targets = input_ids[:, 1:].contiguous().view(-1)
            loss = loss_func(logits, targets)
            
            # back
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                pad_masks = batch['attention_mask'].to(device)
                
                outputs = model(images, input_ids[:, :-1],pad_masks[:,:-1])
                logits = outputs.view(-1, outputs.size(-1))
                targets = input_ids[:, 1:].contiguous().view(-1)
                val_loss += loss_func(logits, targets).item()
        
        # compute bleu score:
        bleu_score_val = evaluateBLEU(model, val_loader)

        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = train_loss/len(train_loader)

        val_bleu.append(bleu_score_val)
        val_losses.append(avg_val_loss)
        train_losses.append(avg_train_loss)

        if use_scheduler:
            scheduler.step(bleu_score_val)
        
        # save best model
        if bleu_score_val > best_bleu:
            best_bleu = bleu_score_val
            torch.save(model.state_dict(), f'best_model_{name}.pth')
            print(f"best model saved: validation bleu score: {best_bleu}")
            
        
        print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val BLEU: {bleu_score_val:.4f}')
    return train_losses, val_losses, val_bleu