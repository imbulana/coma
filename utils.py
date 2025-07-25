import json
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from miditok.pytorch_data import DatasetMIDI

from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import *

# data utils

def group_by_composition(midi_paths, get_title_fn=lambda x: x.parent.name):
    df = pd.DataFrame(midi_paths, columns=['path'])
    df['composition'] = df['path'].apply(get_title_fn)
    return df.groupby('composition')['path'].apply(list).tolist()

class CompositionDataset(Dataset):
    def __init__(self, midi_paths_grouped, **kwargs):
        super().__init__()
        self.groups = [
            DatasetMIDI(paths, **kwargs)
            for paths in midi_paths_grouped
        ]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]

class CompositionDataLoader:
    def __init__(self, grouped_dataset, collator, shuffle=False):
        self.grouped_dataset = grouped_dataset
        self.collator = collator
        self.shuffle = shuffle
        self.group_loaders = [
            DataLoader(
                group,
                batch_size=len(group),
                collate_fn=self.collator,
                shuffle=self.shuffle
            )
            for group in self.grouped_dataset.groups
        ]

    def __len__(self):
        return len(self.group_loaders)

    def __getitem__(self, idx):
        return self.group_loaders[idx]

# train & eval utils

def save_config(writer):
    config_dict = {
        "SEED": SEED,
        "DEVICE": str(DEVICE),
        "SPLIT_DATA": SPLIT_DATA,
        "SHUFFLE": SHUFFLE,
        "TEST_SIZE": TEST_SIZE,
        "TOP_K_COMPOSERS": TOP_K_COMPOSERS,
        "TO_SKIP": TO_SKIP,
        "AUGMENT_DATA": AUGMENT_DATA,
        "TRAIN_TOKENIZER": TRAIN_TOKENIZER,
        "VOCAB_SIZE": VOCAB_SIZE,
        "BEAT_RES": str(BEAT_RES),
        "TOKENIZER_PARAMS": {k: str(v) for k, v in TOKENIZER_PARAMS.items()},
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LR_SCHEDULER": LR_SCHEDULER,
        "MILESTONES": MILESTONES,
        "MAX_SEQ_LEN": MAX_SEQ_LEN,
        "MODEL_PARAMS": {
            "DIM": DIM,
            "DEPTH": DEPTH,
            "DIM_HEAD": DIM_HEAD,
            "HEADS": HEADS,
            "FF_MULT": FF_MULT,
            "ATTN_WINDOW_SIZES": ATTN_WINDOW_SIZES,
            "CONV_EXPANSION_FACTOR": CONV_EXPANSION_FACTOR,
            "CONV_KERNEL_SIZE": CONV_KERNEL_SIZE,
            "ATTN_DROPOUT": ATTN_DROPOUT,
            "FF_DROPOUT": FF_DROPOUT,
            "CONV_DROPOUT": CONV_DROPOUT,
            "PRENORM": PRENORM,
            "QK_SCALE": QK_SCALE,
            "POOLING_STRATEGY": POOLING_STRATEGY,
        }
    }
    writer.add_text("config", json.dumps(config_dict, indent=2))


def print_metrics(
    train_loss, train_acc, train_f1, 
    valid_loss, valid_acc, valid_f1, 
    group_valid_acc_majority = None, group_valid_f1_majority = None, 
    group_valid_acc_confidence_vote = None, group_valid_f1_confidence_vote = None
):
    print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.2f}")
    print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}% | Valid F1: {valid_f1:.2f}\n")

    if group_valid_acc_majority is not None:
        print(f"\nComposition Valid Acc (Majority Vote): {group_valid_acc_majority:.2f}% | Composition Valid F1 (Majority Vote): {group_valid_f1_majority:.2f}")
    if group_valid_acc_confidence_vote is not None:
        print(f"Composition Valid Acc (Confidence Vote): {group_valid_acc_confidence_vote:.2f}% | Composition Valid F1 (Confidence Vote): {group_valid_f1_confidence_vote:.2f}\n")
    
def plot_metrics(
    epoch_labels, epoch_predictions, composer_id2name, epoch, 
    save_path, show_plots, eval_type='chunk'
):
    # confusion matrix
    cm = confusion_matrix(epoch_labels, epoch_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    composer_names = composer_id2name.values()

    plt.figure(figsize=(7, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=composer_names, yticklabels=composer_names)
    plt.title(f'Confusion Matrix (Epoch {epoch+1})')
    plt.xlabel('Predicted Composer')
    plt.ylabel('True Composer')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}/cm/{eval_type}/epoch_{epoch}.png')
    if show_plots:
        plt.show()
    plt.close()

    # per-composer f1 scores
    per_composer_f1 = f1_score(epoch_labels, epoch_predictions, average=None)
    plt.figure(figsize=(7, 4))
    sns.barplot(
        y=list(composer_names), x=per_composer_f1, orient="h", hue=list(composer_names), legend=False
    )
    plt.title(f'Per-Composer F1 Scores (Epoch {epoch})')
    plt.xlabel('F1 Score')
    plt.ylabel('Composer')
    plt.xlim(0, 1)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}/f1/{eval_type}/epoch_{epoch}.png')
    if show_plots:
        plt.show()
    plt.close()

def train_epoch(
    model, train_loader, criterion, optimizer, device, epoch, writer, scheduler=None, max_grad_norm=1.
):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    epoch_labels = []
    epoch_predictions = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(progress_bar):
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device).squeeze(-1)
        attention_mask = batch['attention_mask'].to(torch.bool).to(device)
        
        logits = model(inputs, mask=attention_mask)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        with torch.no_grad():
            probs = logits.softmax(dim=-1)
            predicted = probs.argmax(dim=-1)
        total += len(labels)
        correct += (predicted == labels).sum().item()

        total_loss += loss.item()
        current_loss = total_loss / (batch_idx + 1)
        current_acc = 100. * correct / total
        
        progress_bar.set_postfix({
            'loss': current_loss,
            'accuracy': current_acc,
        })

        epoch_labels.extend(labels.cpu().numpy())
        epoch_predictions.extend(predicted.cpu().numpy())

        # log to tensorboard (per batch)
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('train/batch_loss', loss.item(), global_step)
        writer.add_scalar('train/batch_accuracy', 100. * (predicted == labels).sum().item() / len(labels), global_step)
    
    # log epoch metrics
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    epoch_f1 = f1_score(epoch_labels, epoch_predictions, average='macro')

    writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
    writer.add_scalar('train/epoch_accuracy', epoch_acc, epoch)
    writer.add_scalar('train/epoch_f1', epoch_f1, epoch)
    
    # log learning rate and step scheduler

    if scheduler is not None:
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('train/learning_rate', current_lr, epoch)

        scheduler.step()
    
    return epoch_loss, epoch_acc, epoch_f1

def validate_chunks(
    model, valid_loader, criterion, device, epoch, writer, composer_id2name, test=False, 
    save_path=None, show_plots=False, eval_type='chunk'
):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    epoch_labels = []
    epoch_predictions = []
    
    with torch.no_grad():
        progress_bar = tqdm(valid_loader, desc='Validation (chunks)')
        for batch_idx, group in enumerate(progress_bar):
            inputs = group['input_ids'].to(device)
            labels = group['labels'].to(device).squeeze(-1)
            attention_mask = group['attention_mask'].to(torch.bool).to(device)
            
            logits = model(inputs, mask=attention_mask)
            loss = criterion(logits, labels)
            
            probs = logits.softmax(dim=-1)
            predicted = probs.argmax(dim=-1)
            total += len(labels)
            correct += (predicted == labels).sum().item()

            total_loss += loss.item()
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            
            progress_bar.set_postfix({
                'loss': current_loss,
                'accuracy': current_acc,
            })
    
            epoch_labels.extend(labels.cpu().numpy())
            epoch_predictions.extend(predicted.cpu().numpy())

    # epoch metrics
    epoch_loss = total_loss / len(valid_loader)
    epoch_acc = 100. * correct / total
    epoch_f1 = f1_score(epoch_labels, epoch_predictions, average='macro')

    plot_metrics(epoch_labels, epoch_predictions, composer_id2name, epoch, save_path, show_plots, eval_type)

    val_type = 'test' if test else 'val'
    writer.add_scalar(f'{val_type}/chunk/epoch_loss', epoch_loss, epoch)
    writer.add_scalar(f'{val_type}/chunk/epoch_accuracy', epoch_acc, epoch)
    writer.add_scalar(f'{val_type}/chunk/epoch_f1', epoch_f1, epoch)
    
    return epoch_loss, epoch_acc, epoch_f1

def validate_composition(
    model, valid_loader, device, epoch, writer, composer_id2name,
    save_path=None, show_plots=False, eval_type='composition'
):
    model.eval()

    correct_majority = 0
    total_majority = 0
    epoch_labels_majority = []
    epoch_predictions_majority = []

    correct_confidence_vote = 0
    total_confidence_vote = 0
    epoch_labels_confidence_vote = []
    epoch_predictions_confidence_vote = []

    with torch.no_grad():
        progress_bar = tqdm(valid_loader, desc='Validation (composition)')
        for batch_idx, batch in enumerate(progress_bar):
            for group in batch:
                inputs = group['input_ids'].to(device)
                label = group['labels'].to(device)[0]
                attention_mask = group['attention_mask'].to(torch.bool).to(device)
                
                logits = model(inputs, mask=attention_mask)
                probs = logits.softmax(dim=-1)

                confidence_vote_predicted = probs.sum(dim=0).argmax(dim=-1)

                # majority vote

                chunk_predictions = probs.argmax(dim=-1)
                
                # count votes for each composer
                votes = torch.zeros(len(composer_id2name), device=device)
                for i in range(len(composer_id2name)):
                    votes[i] = (chunk_predictions == i).sum()

                predicted = votes.argmax(dim=-1).unsqueeze(-1)

                assert predicted.shape == label.shape

                # update metrics

                total_majority += 1
                correct_majority += (predicted == label).item()
                current_acc_majority = 100. * correct_majority / total_majority

                total_confidence_vote += 1
                correct_confidence_vote += (confidence_vote_predicted == label).item()
                current_acc_confidence_vote = 100. * correct_confidence_vote / total_confidence_vote

                progress_bar.set_postfix({
                    'acc_majority': current_acc_majority,
                    'acc_confidence_vote': current_acc_confidence_vote,
                })
        
                epoch_labels_majority.append(label.item())
                epoch_predictions_majority.append(predicted.item())
                epoch_labels_confidence_vote.append(label.item())
                epoch_predictions_confidence_vote.append(confidence_vote_predicted.item())

    plot_metrics(
        epoch_labels_majority, epoch_predictions_majority, composer_id2name, epoch, 
        save_path, show_plots, eval_type
    )
    plot_metrics(
        epoch_labels_confidence_vote, epoch_predictions_confidence_vote, composer_id2name, epoch, 
        save_path, show_plots, eval_type
    )

    # log epoch metrics

    epoch_acc_majority = 100. * correct_majority / total_majority
    epoch_f1_majority = f1_score(epoch_labels_majority, epoch_predictions_majority, average='macro')
    epoch_acc_confidence_vote = 100. * correct_confidence_vote / total_confidence_vote
    epoch_f1_confidence_vote = f1_score(epoch_labels_confidence_vote, epoch_predictions_confidence_vote, average='macro')

    writer.add_scalar('val/composition/epoch_accuracy_majority', epoch_acc_majority, epoch)
    writer.add_scalar('val/composition/epoch_f1_majority', epoch_f1_majority, epoch)
    writer.add_scalar('val/composition/epoch_accuracy_confidence_vote', epoch_acc_confidence_vote, epoch)
    writer.add_scalar('val/composition/epoch_f1_confidence_vote', epoch_f1_confidence_vote, epoch)
    
    return epoch_acc_majority, epoch_f1_majority, epoch_acc_confidence_vote, epoch_f1_confidence_vote
