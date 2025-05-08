import torch
from tqdm import tqdm
from sacrebleu import corpus_bleu
from config import MAX_LENGTH
from data import tokenizer




def generate_captions(model, dataloader, tokenizer, max_length):
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            
            pred_ids = model.generate(images, max_length=max_length, tokenizer=tokenizer).cpu() 
            
            references = [
                tokenizer.decode(ids, skip_special_tokens=True) 
                for ids in batch['input_ids'].numpy()
            ]
            
            predictions = [
                tokenizer.decode(ids, skip_special_tokens=True) 
                for ids in pred_ids
            ]

            all_preds.extend(predictions)
            all_targets.extend(references)
    
    return all_preds, all_targets

def evaluateBLEU(model, dataloader):
    all_preds, all_targets = generate_captions(model, dataloader, tokenizer=tokenizer, max_length=MAX_LENGTH)

    references = [[ref] for ref in all_targets]  # 注意这里不split，保持完整字符串
    hypotheses = all_preds  # 也保持完整字符串
    
    bleu_score = corpus_bleu(hypotheses, references)
    
    print(f"BLEU Score for the model: {bleu_score.score}")
    
    return bleu_score.score