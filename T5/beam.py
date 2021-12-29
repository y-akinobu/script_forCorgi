
#インポート
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    get_linear_schedule_with_warmup
)
import textwrap
from tqdm.auto import tqdm
import shutil
import argparse
import time
import datetime

# オプション
parser = argparse.ArgumentParser(description='Using mT5 with ABCI')
parser.add_argument('input_file', help='corpus')
parser.add_argument('--model', action='store_true', help='Save the model as a zip file')
parser.add_argument('--result', action='store_true', help='Output the result as a tsv file')

args = parser.parse_args()

d_now = datetime.datetime.now()


# 事前学習済みモデル
PRETRAINED_MODEL_NAME = "google/mt5-small"

# 転移学習済みモデルを保存する場所
DATA_DIR = "data"
MODEL_DIR = "model"

# INPUT file
INPUT_tsv = args.input_file

# GPU利用有無
USE_GPU = torch.cuda.is_available()

def from_tsv(f_path):
  df = pd.read_table(f_path, header=None)

  df = df.dropna()

  df = df.drop_duplicates()
  df = df.reset_index(drop=True)

  df = df.sample(frac=1, random_state=0).reset_index(drop=True)

  df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
  df_valid, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

  df_train.to_csv('data/train.tsv', header=False, index=False, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, escapechar='\n')
  df_valid.to_csv('data/dev.tsv', header=False, index=False, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, escapechar='\n')
  df_test.to_csv('data/test.tsv', header=False, index=False, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, escapechar='\n')

  return len(df_train), len(df_valid), len(df_test)


# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TsvDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, input_max_len=512, target_max_len=512):
        self.file_path = os.path.join(data_dir, type_path)
        
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        
        self._build()
  
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _make_record(self, src, tgt):
        input = f"{src}"
        target = f"{tgt}"
        return input, target
  
    def _build(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split("\t")
                assert len(line[0]) > 0
                assert len(line[1]) > 0

                tgt = line[0]
                src = line[1]

                input, target = self._make_record(src, tgt)

                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input], max_length=self.input_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target], max_length=self.target_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)


class MT5FineTuner(pl.LightningModule):
    def __init__(self, 
                 data_dir=DATA_DIR,
                 model_name_or_path=PRETRAINED_MODEL_NAME,
                 tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

                 learning_rate=3e-4,
                 weight_decay=0.0,
                 adam_epsilon=1e-8,
                 warmup_steps=0,
                 gradient_accumulation_steps=1,

                 max_input_length=128,
                 max_target_length=128,
                 train_batch_size=4,
                 eval_batch_size=4,
                 num_train_epochs=4,

                 n_gpu=1 if USE_GPU else 0,
                 early_stop_callback=False,
                 fp_16=False,
                 opt_level='O2',
                 max_grad_norm=1.0,
                 seed=42,
                 ):
        super().__init__()
        self.save_hyperparameters()

        # 事前学習済みモデルの読み込み
        self.model = MT5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)

        # トークナイザーの読み込み
        self.tokenizer = MT5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path, is_fast=True)
        self.tokenizer.add_tokens(additional_special_tokens)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]

        # All labels set to -100 are ignored (masked), 
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    # def validation_epoch_end(self, outputs):
    #     """バリデーション完了処理"""
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     self.log("val_loss", avg_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    # def test_epoch_end(self, outputs):
    #     """テスト完了処理"""
    #     avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     self.log("test_loss", avg_loss, prog_bar=True)

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.hparams.learning_rate, 
                          eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_dataset(self, tokenizer, type_path, args):
        """データセットを作成する"""
        return TsvDataset(
            tokenizer=tokenizer, 
            data_dir=args.data_dir, 
            type_path=type_path, 
            input_max_len=args.max_input_length,
            target_max_len=args.max_target_length)
    
    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                             type_path="train.tsv", args=self.hparams)
            self.train_dataset = train_dataset

            val_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                           type_path="dev.tsv", args=self.hparams)
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
            )

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset, 
                          batch_size=self.hparams.train_batch_size, 
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset, 
                          batch_size=self.hparams.eval_batch_size, 
                          num_workers=4) 


#プログラムの実行
if __name__ == '__main__':
    train_num, valid_num, test_num = from_tsv(INPUT_tsv)

    set_seed(42)

    var_tokens = []
    name_val_tokens = []

    for idx in range(1, 7):
        var_tokens.append(f'<var{idx}>')
        name_val_tokens.append(f'<name{idx}>')
        name_val_tokens.append(f'<val{idx}>')

    # パラメータ化用の特殊トークン
    additional_special_tokens = ['<A>', '<B>', '<C>', '<D>', '<E>', '<a>', '<b>', '<c>', '<d>', '<e>']
    # additional_special_tokens = name_val_tokens

    # トークナイザー（SentencePiece）モデルの読み込み
    tokenizer = MT5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME, is_fast=True)
    tokenizer.add_tokens(additional_special_tokens)

    # テストデータセットの読み込み
    test_dataset = TsvDataset(tokenizer, DATA_DIR, "test.tsv", 
                            input_max_len=128, target_max_len=128)

    train_params = dict(
        accumulate_grad_batches = 1,
        gpus = 1 if USE_GPU else 0,
        max_epochs = 50,
        precision= 32,
        gradient_clip_val=1.0,
        # amp_level=args.opt_level,
        # checkpoint_callback=checkpoint_callback,
    )

    # 転移学習の実行（GPUを利用すれば1エポック10分程度）
    s1 = time.time()
    model = MT5FineTuner()
    trainer = pl.Trainer(**train_params, callbacks=[EarlyStopping(monitor="val_loss")])
    trainer.fit(model)
    s2 = time.time()

    # 最終エポックのモデルを保存
    model.tokenizer.save_pretrained(MODEL_DIR)
    model.model.save_pretrained(MODEL_DIR)

    del model

    # トークナイザー（SentencePiece）
    tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True)

    # 学習済みモデル
    trained_model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    # GPUの利用有無
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        trained_model.cuda()


    # 学習済みモデルのパラメータ数
    params = 0
    for p in trained_model.parameters():
        if p.requires_grad:
            params += p.numel()

    # テストデータの読み込み
    test_dataset = TsvDataset(tokenizer, DATA_DIR, "test.tsv", 
                            input_max_len=128, 
                            target_max_len=128)

    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4)

    trained_model.eval()
    trained_model.config.update({"num_beams": 5})

    outputs = []
    targets = []
    sources = []

    for batch in tqdm(test_loader):
        input_ids = batch['source_ids']
        input_mask = batch['source_mask']

        if USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        outs = trained_model.generate(
            input_ids=input_ids, 
            attention_mask=input_mask, 
            max_length=64,
            num_beams=5,
            # length_penalty=8.0,
            no_repeat_ngram_size=2,
            num_return_sequences=5, 
            return_dict_in_generate=True,
            output_scores=True,
            early_stopping=True
            )

        cnt_beams = 0
        pred = []
        prob = []
        dec = []
        
        for ids, s in zip(outs.sequences, outs.sequences_scores):
            if cnt_beams < 5:
                pred.append(ids)
                prob.append(s)
                cnt_beams += 1
            else:
                ids_top1 = pred[np.argmax(prob)]
                dec.append(tokenizer.decode(ids_top1, skip_special_tokens=True, 
                                clean_up_tokenization_spaces=False))
                cnt_beams = 0
                pred = []
                prob = []

        target = [tokenizer.decode(ids, skip_special_tokens=True, 
                                   clean_up_tokenization_spaces=False) 
                    for ids in batch["target_ids"]]
        source = [tokenizer.decode(ids, skip_special_tokens=True, 
                                   clean_up_tokenization_spaces=False) 
                    for ids in batch["source_ids"]]

        outputs.extend(dec)
        targets.extend(target)
        sources.extend(source)


    df = pd.DataFrame(list(zip(sources, targets, outputs)), columns = ['JPN', 'Py', ' Predict'])

    INPUT_tsv = INPUT_tsv[INPUT_tsv.rfind('/')+1:]
    INPUT_tsv_name = INPUT_tsv.replace('.tsv', '')
    OUTPUT_tsv = f'result_mT5_{INPUT_tsv_name}.tsv'

    if args.model:
        shutil.make_archive(f'trained_model/model_mT5_{INPUT_tsv_name}', 'zip', root_dir='model')
    elif args.result:
        df.to_csv(f'result/{OUTPUT_tsv}', index=False, header=False, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, escapechar='\n')
    else:
        df.to_csv(f'result/{OUTPUT_tsv}', index=False, header=False, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, escapechar='\n')
        shutil.make_archive(f'trained_model/model_mT5_{INPUT_tsv_name}', 'zip', root_dir='model')

    with open('log/trainlog_mT5.txt', mode= 'a') as f:
        f.write(d_now.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write(f'Input File : {INPUT_tsv}\n')
        f.write(f'Output File : result/{OUTPUT_tsv}\n')
        f.write(f'Pretrained Model : trained_model/model_mT5_{INPUT_tsv_name}.zip\n')
        f.write(f'Train Data : {train_num}, Validation Data : {valid_num}, Test Data : {test_num}\n')
        f.write(f'Number of parameters : {params}\n')
        f.write(f'Training Time : {s2-s1}\n')
        f.write('\n')

    print('--- FINISH ---')
    print('INPUT :', INPUT_tsv)
    print(f'Train Data : {train_num}, Validation Data : {valid_num}, Test Data : {test_num}')
    print('Number of parameters :', params)
    print('Training Time :', s2-s1)
