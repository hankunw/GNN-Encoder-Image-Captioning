import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pycocotools.coco import COCO
import os

from config import MAX_LENGTH, BATCH_SIZE
from utils import tokenizer, transform, VOCAB_SIZE, parse_captions, split_dataset, collate_fn


class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions_dict, img_list, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        # 构建样本列表 (图片路径, caption)
        for img_name in img_list:
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):  # 确保图片存在
                for caption in captions_dict.get(img_name, []):
                    self.samples.append((img_path, caption))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, caption
    



class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file, max_length, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.max_length = max_length

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Get image path
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get the first caption (you can modify this for multiple captions)
        caption = annotations[0]['caption'] if annotations else ""

        encoding = tokenizer(
            caption, 
            max_length = self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        attention_mask = encoding['attention_mask'].squeeze()
        pad_mask = (attention_mask == 0)
        return {
            'image':image,
            'input_ids':encoding['input_ids'].squeeze(),
            'attention_mask': pad_mask
        }
