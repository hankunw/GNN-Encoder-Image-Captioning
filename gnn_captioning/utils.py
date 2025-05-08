import random
import math
from collections import defaultdict
import torch
from transformers import GPT2Tokenizer
from torchvision import transforms


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def parse_captions(txt_path):
    '''
        input: caption path
        return: a dictionary, key: image names; value: caption
    '''
    captions_dict = defaultdict(list)
    with open(txt_path, 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 假设每行格式：图片名\tcaption
            parts = line.split(',')
            if len(parts) >= 2:
                img_name, caption = parts[0], parts[1]
                captions_dict[img_name].append(caption)
    return captions_dict

def split_dataset(captions_dict, ratios=(0.6, 0.2, 0.2), seed=42):
    '''
        split dataset with ratio
        input ratio sample: (0.6,0.2,0.2)
        return: (train, val,test)
    '''
    assert sum(ratios) == 1.0, "sum of ratios must be one"
    img_names = list(captions_dict.keys())
    random.seed(seed)
    random.shuffle(img_names)
    
    total = len(img_names)
    train_end = int(ratios[0] * total)
    val_end = train_end + int(ratios[1] * total)
    
    return (
        img_names[:train_end],
        img_names[train_end:val_end],
        img_names[val_end:]
    )

def collate_fn(batch, m_len):
    """自定义批次处理函数"""
    images, texts = zip(*batch)
    
    # 图像处理
    images = torch.stack(images, dim=0)
    
    # 文本编码
    text_encodings = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=m_len,
        return_tensors='pt',
        add_special_tokens=True

    )
    attention_mask = text_encodings['attention_mask'].squeeze()
    pad_mask = (attention_mask == 0)
    return {
        'image': images,
        'input_ids': text_encodings['input_ids'],
        'attention_mask': pad_mask
    }    