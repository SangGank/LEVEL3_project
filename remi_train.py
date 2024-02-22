from miditok import MMM, MuMIDI, TokenizerConfig, Octuple, REMI
from miditok.pytorch_data import DatasetTok, DataCollator
from pathlib import Path
from symusic import Score
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import torch

from transformers import AutoConfig, GPT2LMHeadModel
import wandb

wandb.init(project="LEVEL3", entity="sanggang", name='REMI')


class MidiDataset(Dataset):
    def __init__(self, files_paths, label_path = None, max_length=1022, tokenizer = REMI):  # max_length를 512로 하면 앞, 뒤에 BOS, EOS 토큰이 또 붙어서 길이 514 되고 에러가 나서 일단 510로 함. 디버깅 필요!!

        # self.tokenized_songs = { 'input_ids' : [], 'attention_mask' : [], 'label' : [] }
        self.tokenized_songs = []
        self.labels = []
        self.tokenizer = tokenizer

        for song_path in tqdm(files_paths):
            song = self.tokenizer(song_path).ids[:max_length]
            song_length = len(song)
            attention = [1] * song_length
            # 
            if song_length < max_length:
                song += [0] * (max_length - song_length)
                attention += [0] * (max_length - song_length)
            dic = {'input_ids': torch.tensor(song),  
                'attention_mask': torch.tensor(attention)}
            self.tokenized_songs.append(dic)

            # self.tokenized_songs['input_ids'].append(torch.tensor(song))
            # self.tokenized_songs['attention_mask'].append(torch.tensor(attention))
            
            if label_path:

                continue

            else:
                self.labels.append(torch.tensor(song))

    def __len__(self):
        return len(self.tokenized_songs)
    
    def __getitem__(self, idx):

        return {
            'input_ids': self.tokenized_songs[idx]['input_ids'],  
            'attention_mask': self.tokenized_songs[idx]['attention_mask'],
            'labels': self.labels[idx]
        }
    

    


TOKENIZER_PARAMS = {
    "pitch_range": (0, 127),
    "num_velocities": 127,
    # "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    # "max_bar_embedding" : 3000,
    "use_chords": True,
    "use_tempos": True,
    "use_programs": True,
    "num_tempos": 211,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}
config = TokenizerConfig(**TOKENIZER_PARAMS)
TOKENIZER_NAME = REMI
tokenizer = TOKENIZER_NAME(config)


collator = DataCollator(
    tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"], labels_pad_idx = tokenizer["PAD_None"]
)


midi_paths = list(Path("jazz_max_token").glob("**/*.mid"))

total_num_files = len(midi_paths)
num_files_valid = round(total_num_files * 0.1) # Validation 비율 자유롭게 변경
midi_paths=sorted(midi_paths)
midi_paths_valid = midi_paths[:num_files_valid]
midi_paths_train = midi_paths[num_files_valid:]



# dataset_train = DatasetTok(
#     files_paths=midi_paths_train,
#     min_seq_len=50,
#     max_seq_len=1022,
#     tokenizer=tokenizer,
# )
# dataset_valid = DatasetTok(
#     files_paths=midi_paths_valid,
#     min_seq_len=50,
#     max_seq_len=1022,
#     tokenizer=tokenizer,
# )


dataset_train = MidiDataset(files_paths = midi_paths_train, tokenizer = tokenizer)
dataset_valid = MidiDataset(files_paths = midi_paths_valid, tokenizer = tokenizer)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # call super class method to get the eval outputs
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        return eval_output
    

n_layer=6
n_head=4
n_emb=1024
context_length = 1024

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_positions=context_length,
    n_layer=n_layer,
    n_head=n_head,
    pad_token_id=tokenizer["PAD_None"],
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
    n_embd=n_emb
)

model = GPT2LMHeadModel(config)
output_path = "model_none"
steps = 200
# Commented parameters correspond to the small model
config = {"output_dir": output_path,
          "num_train_epochs": 30, # 학습 epoch 자유롭게 변경. 저는 30 epoch 걸어놓고 early stopping 했습니다.
          "per_device_train_batch_size": 16,
          "per_device_eval_batch_size": 16,
          "evaluation_strategy": "steps",
          "save_strategy": "steps",
          "eval_steps": steps,
          "logging_steps":steps,
          "logging_first_step": True,
          "save_total_limit": 5,
          "save_steps": steps,
          "lr_scheduler_type": "cosine",
          "learning_rate":5e-4,
          "warmup_ratio": 0.01,
          "weight_decay": 0.01,
          "seed": 1,
          "load_best_model_at_end": True,
          "metric_for_best_model": "eval_loss", # best model 기준 바꾸고 싶을 경우 이 부분 변경 (default가 eval_loss임)
          "report_to": "wandb"
          }



# mps device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_args = TrainingArguments(**config)

trainer = CustomTrainer(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    data_collator=collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)] # Early Stopping patience 자유롭게 변경
)
print(device)


# Train the model.
trainer.train()

