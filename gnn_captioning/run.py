import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import argparse

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import train_model
from config import BATCH_SIZE, MAX_LENGTH, flicker_path, flicker_caption, val_coco_dir, coco_val_annos, flicker_ratio, LEARNING_RATE, device
from data import Flickr8kDataset, COCODataset, VOCAB_SIZE, tokenizer, transform
from evaluate import evaluateBLEU
from utils import parse_captions, split_dataset, collate_fn


from model.gcn_encoder import GCN_Encoder
from model.gat_encoder import GAT_Encoder
from model.vit_encoder import ViT_Encoder
from model.decoder import ImageCaptioningModel

def get_model(model_name):
    if model_name == "gcn":
        encoder = GCN_Encoder()
    elif model_name == "gat":
        encoder = GAT_Encoder()
    elif model_name == "vit":
        encoder = ViT_Encoder()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = ImageCaptioningModel(encoder=encoder, vocab_size=VOCAB_SIZE, max_seq_len=MAX_LENGTH)
    return model, model_name

def get_flicker_dataloader(image_path, caption_path, batch_size = BATCH_SIZE):
    
    
    captions_dict = parse_captions(caption_path)
    train_list, val_list, test_list = split_dataset(captions_dict, ratios=flicker_ratio)
    train_dataset = Flickr8kDataset(
        image_path, captions_dict, train_list, transform=transform
    )
    val_dataset = Flickr8kDataset(
        image_path, captions_dict, val_list, transform=transform
    )
    test_dataset = Flickr8kDataset(
        image_path, captions_dict, test_list, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader

def get_coco_dataloader(image_path, annotation, batch_size = BATCH_SIZE):
    

    coco_test_dataset = COCODataset(image_path, annotation, max_length=MAX_LENGTH,transform=transform)

    coco_test_loader = DataLoader(coco_test_dataset, batch_size=batch_size, shuffle=False)
    return coco_test_loader

def plot_curve(train_losses, val_losses, val_bleu):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1,len(train_losses)+1), train_losses,label='train', marker='o', color='blue')
    plt.plot(range(1,len(train_losses)+1), val_losses, label='val', marker='x', color='green')
    plt.title("Training Loss and Val Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(8, 5))
    plt.plot(range(1,len(train_losses)+1), val_bleu, marker='o', color='blue')
    # plt.plot(range(1,len(train_losses)+1), val_losses, label='val', marker='x', color='green')
    plt.title("BLEU score on Validation Set")
    plt.xlabel('Epochs')
    plt.ylabel('BLEU score')
    plt.legend()
    plt.grid(True)
    plt.show()

def reload_model(model, pth):
    model_path = pth
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate/")

    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate'],
                        help="Mode: train / evaluate ")
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'gat', 'vit'],
                        help="Model type: please choose one from gcn, gat and vit")
    parser.add_argument('--testset', type=str, default='coco',
                        choices=['flicker', 'coco'],
                        help=" Test Dataset Choice, coco and flicker")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to model checkpoint for evaluation")

    args = parser.parse_args()

    # Load data
    flicker_train_loader, flicker_val_loader, flicker_test_loader = get_flicker_dataloader(flicker_path, flicker_caption)
    coco_test_loader = get_coco_dataloader(val_coco_dir,coco_val_annos)

    if args.testset == 'flicker':
        test_loader = flicker_test_loader
    elif args.testset == 'coco':
        test_loader = coco_test_loader
    else:
        raise ValueError(f"Unknown test set: {args.testset}")

    # Build model
    model, name = get_model(args.model)


    # initialize loss_func and optimizer
    loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # train
    if args.mode == "train":
        train_losses, val_losses, val_bleu_scores=train_model(model, name, flicker_train_loader, flicker_val_loader, loss_func, optimizer)
        # TODO: add training curve
        plot_curve(train_losses, val_losses, val_bleu_scores)

    elif args.mode == "evaluate":
        reloaded_model = reload_model(model, args.checkpoint)
        eval_bleu = evaluateBLEU(reloaded_model, test_loader)
        print(f"BLEU score for captioning model with {name} encoder is {eval_bleu}")
    # elif args.mode == "inference":
    #     run_inference(encoder, dec, test_loader, args.checkpoint, cfg)

if __name__ == "__main__":
    main()