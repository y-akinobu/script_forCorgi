#import
import torch
import torchtext
import torch.nn as nn
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.utils import unicode_csv_reader
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torch import Tensor
from typing import Iterable, List
# import sentencepiece as spm
from janome.tokenizer import Tokenizer
import numpy as np
import pandas as pd
import csv
import io
import math
from sklearn.model_selection import train_test_split, KFold
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
# from sklearn.model_selection import cross_val_score
from timeit import default_timer as timer
import shutil
import argparse
import datetime
import time
import os

# オプションを設定する
parser = argparse.ArgumentParser(description='Using Transformer with ABCI')
parser.add_argument('input_file', help='corpus')
# parser.add_argument('-e','--epochs', help='epochs',type=int,default=50)
parser.add_argument('--model', action='store_true', help='Save the model as a zip file')
parser.add_argument('--result', action='store_true', help='Output the result as a tsv file')

args = parser.parse_args()

d_now = datetime.datetime.now()


# 関数の定義

def jpn_tokenizer(text):
  return [token for token in tokenizer.tokenize(text) if token != " " and len(token) != 0]

def py_tokenizer(text):
  return [tok for tok in text.split()][:MAX_LEN]

# tsv ファイルから train/valid/test それぞれのファイルを作成する関数
def from_tsv(f_path):
  df = pd.read_table(f_path, header=None)
  # 欠損値の削除
  df = df.dropna()
  print('BEFORE', df.shape)

  # 重複データの削除
  df.drop_duplicates(inplace=True)
  df = df.reset_index(drop=True)
  print('AFTER_drop', df.shape)

  # データ数の調整
  # df = df.iloc[range(1347), :]
  # print('AFTER_iloc', df.shape)

  df = df.sample(frac=1, random_state=0).reset_index(drop=True)

  df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
  df_valid, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

  df_train.to_csv('data/train.tsv', header=False, index=False, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, escapechar='\n')
  df_valid.to_csv('data/valid.tsv', header=False, index=False, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, escapechar='\n')
  df_test.to_csv('data/test.tsv', header=False, index=False, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, escapechar='\n')

  NUM_LINES = {
      'train': len(df_train),
      'valid': len(df_valid),
      'test': len(df_test),
  }

  return NUM_LINES

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

def _create_data_from_tsv(data_path):
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f, delimiter='\t')
        for row in reader:
            yield row[1], row[0]   # SRC/TGT の順でreturn?

def JPN2Py(root, split, language_pair=('jpn', 'py')):
    path = root + '/' + split + '.tsv'

    # src とtgt をセットにしたデータセットを返す
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split], _create_data_from_tsv(path))

class Seq2SeqTransformer(nn.Module):
    def __init__(self, 
                 num_encoder_layers: int, 
                 num_decoder_layers: int,
                 emb_size: int, 
                 nhead: int, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, 
                src: Tensor, 
                tgt: Tensor, 
                src_mask: Tensor,
                tgt_mask: Tensor, 
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, 
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, 
                 emb_size: int, 
                 dropout: float, 
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + 
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([SOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

def train_epoch(model, optimizer):
    model.train()
    losses = 0

    # 学習データ
    train_iter = JPN2Py(root='data', split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    # 検証用データ
    val_iter = JPN2Py(root='data', split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        # logits = cross_val_score(logits, src, tgt_input, cv=kf)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


def beam_topk(model, ys, memory, beamsize):
    ys = ys.to(DEVICE)

    tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
    out = model.decode(ys, memory, tgt_mask)
    out = out.transpose(0, 1)
    prob = model.generator(out[:, -1])
    next_prob, next_word = prob.topk(k=beamsize, dim=1)
    
    return next_prob, next_word

def beam_decode(model, src, src_mask, max_len, beamsize, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    ys_result = {}

    memory = model.encode(src, src_mask).to(DEVICE)   # encode の出力 (コンテキストベクトル)

    # 初期値 (beamsize)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    next_prob, next_word = beam_topk(model, ys, memory, beamsize)
    next_prob = next_prob[0].tolist()

    # <sos> + 1文字目 の候補 (list の長さはbeamsizeの数)
    ys = [torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word[:, idx].item())], dim=0) for idx in range(beamsize)]

    for i in range(max_len-1):
        prob_list = []
        ys_list = []

        # それぞれの候補ごとに次の予測トークンとその確率を計算
        for ys_token in ys:
            next_prob, next_word = beam_topk(model, ys_token, memory, len(ys))

            # 予測確率をリスト (next_prob) に代入
            next_prob = next_prob[0].tolist()
            # 1つのリストに結合
            prob_list.extend(next_prob)

            ys = [torch.cat([ys_token, torch.ones(1, 1).type_as(src.data).fill_(next_word[:, idx].item())], dim=0) for idx in range(len(ys))]
            ys_list.extend(ys)

        # prob_list の topk のインデックスを prob_topk_idx で保持
        prob_topk_idx = list(reversed(np.argsort(prob_list).tolist()))
        prob_topk_idx = prob_topk_idx[:len(ys)]
        # print('@@', prob_topk_idx)

        # ys に新たな topk 候補を代入
        ys = [ys_list[idx] for idx in prob_topk_idx]

        next_prob = [prob_list[idx] for idx in prob_topk_idx]
        # print('@@orig', prob_list)
        # print('@@next', next_prob)

        pop_list = []
        for j in range(len(ys)):
            # EOS トークンが末尾にあったら、ys_result (返り値) に append
            if ys[j][-1].item() == EOS_IDX:
                ys_result[ys[j]] = next_prob[j]
                pop_list.append(j)

        # ys_result に一度入ったら、もとの ys からは抜いておく
        # (ys の長さが変わるので、ところどころbeamsize ではなく len(ys) を使用している箇所がある)
        for l in sorted(pop_list, reverse=True):
            del ys[l]

        # ys_result が beamsize よりも大きかった時に、処理を終える
        if len(ys_result) >= beamsize:
            break

    return ys_result

def translate(model: torch.nn.Module, src_sentence: str, beamsize):
    pred_list = []
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = beam_decode(
        model,  src, src_mask, max_len=num_tokens + 5, beamsize=beamsize, start_symbol=SOS_IDX)
    prob_list = list(tgt_tokens.values())
    tgt_tokens = list(tgt_tokens.keys())
    for idx in list(reversed(np.argsort(prob_list).tolist())):
        pred_list.append(" ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens[idx].cpu().numpy()))).replace("<sos>", "").replace("<eos>", ""))
    return pred_list, sorted(prob_list, reverse=True)


if __name__ == '__main__':

    # 再現性のためにSEED 値を固定
    SEED = 1234

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    INPUT_tsv = args.input_file
    BEAMSIZE = 5

    # SRC (source) : 原文
    SRC_LANGUAGE = 'jpn'
    # TGT (target) : 訳文
    TGT_LANGUAGE = 'py'
    0
    # Place-holders
    token_transform = {}
    vocab_transform = {}

    # トークナイザの定義
    MAX_LEN=80
    tokenizer = Tokenizer("user_simpledic.csv", udic_type="simpledic", udic_enc="utf8", wakati=True)

    # Place-holder に各トークナイザを格納
    token_transform[SRC_LANGUAGE] = jpn_tokenizer
    token_transform[TGT_LANGUAGE] = py_tokenizer

    NUM_LINES = from_tsv(INPUT_tsv)
    
    # 特殊トークンの定義
    UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
    # special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>', '<blk>', '</blk>', '<sep>', '<A>', '<B>', '<C>', '<D>', '<E>', '<F>', '<G>', '<H>', '<I>', '<J>', '<K>', '<L>', '<M>', '<N>', '<O>', '<P>', '<Q>', '<R>', '<S>', '<T>', '<U>', '<V>', '<W>', '<X>', '<Y>', '<Z>', '<a>', '<b>', '<c>', '<d>', '<e>', '<f>', '<g>', '<h>', '<i>', '<j>', '<k>', '<l>', '<m>', '<n>', '<o>', '<p>', '<q>', '<r>', '<s>', '<t>', '<u>', '<v>', '<w>', '<x>', '<y>', '<z>']
    special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>', '<blk>', '</blk>', '<sep>', '<A>', '<B>', '<C>', '<D>', '<E>', '<a>', '<b>', '<c>', '<d>', '<e>', '<var1>', '<var2>', '<var3>', '<var4>', '<var5>', '<var6>', '<var7>', '<val1>', '<val2>', '<val3>', '<val4>', '<val5>', '<val6>', '<val7>', '<name1>', '<name2>', '<name3>', '<name4>', '<name5>', '<name6>', '<name7>']

    DATASET_NAME = "JPN2Py"

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # train data の IterableDataset を作成
        train_iter = JPN2Py(root='data', split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  
        # ボキャブラリの作成 (ボキャブオブジェクト)
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

    # トークンが見つからなかったときに返す <unk> トークンを定義
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
      vocab_transform[ln].set_default_index(UNK_IDX)

    # パラメータの定義
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 32
    # BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    # デバイスの指定
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # インスタンスの作成
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
                                     EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                     FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # デバイスの設定
    transformer = transformer.to(DEVICE)

    # 損失関数の定義 (クロスエントロピー)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # オプティマイザの定義 (Adam)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    
    # 文字列をテンソルのインデックスに変換するための言語テキスト変換
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                   vocab_transform[ln], #Numericalization
                                                   tensor_transform) # Add SOS/EOS and create tensor

    NUM_EPOCHS = 50
    CNT = 0

    best_val_loss = float('inf')

    s1 = time.time()
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(transformer.state_dict(), 'tut6-model.pt')
            CNT = 0
        else:
            CNT += 1

        if CNT == 3:
            print('Early Stopping')
            break

        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    transformer.load_state_dict(torch.load('tut6-model.pt'))
    s2 = time.time()

    # 学習済みモデルのパラメータ数
    params = 0
    for p in transformer.parameters():
        if p.requires_grad:
            params += p.numel()

    test_iter = JPN2Py(root='data', split='test', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

    cols = ['source', 'target', 'pred']
    df = pd.DataFrame(index=[], columns=cols)

    for test_sentence in test_iter:
        pred, _ = translate(transformer, test_sentence[0], BEAMSIZE)
        if len(pred) != 0:
            df = df.append({'source': test_sentence[0], 'target': test_sentence[1].strip(), 'pred': pred[0].strip()}, ignore_index=True)

    INPUT_tsv = INPUT_tsv[INPUT_tsv.rfind('/')+1:]
    INPUT_tsv_name = INPUT_tsv.replace('.tsv', '')
    OUTPUT_tsv = f'result_TF_{INPUT_tsv_name}.tsv'

    if args.result:
        df.to_csv(f'result/{OUTPUT_tsv}', index=False, header=False, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, escapechar='\n')
    elif args.model:
        os.mkdir(f'trained_model/{INPUT_tsv}_forTF')
        torch.save(vocab_transform, f'trained_model/{INPUT_tsv}_forTF/vocab_obj.pth')
        torch.save(transformer.state_dict(), f'trained_model/{INPUT_tsv}_forTF/model.pt')
        shutil.make_archive(f'trained_model/{INPUT_tsv}_forTF_p','zip',root_dir=f'trained_model/{INPUT_tsv}_forTF')
    else:
        df.to_csv(f'result/{OUTPUT_tsv}', index=False, header=False, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, escapechar='\n')
        os.mkdir(f'trained_model/{INPUT_tsv}_forTF')
        torch.save(vocab_transform, f'trained_model/{INPUT_tsv}_forTF/vocab_obj.pth')
        torch.save(transformer.state_dict(), f'trained_model/{INPUT_tsv}_forTF/model.pt')
        shutil.make_archive(f'trained_model/{INPUT_tsv}_forTF_p','zip',root_dir=f'trained_model/{INPUT_tsv}_forTF')

    with open('log/trainlog_TF.txt', mode= 'a') as f:
        f.write(d_now.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write(f'Input File : {INPUT_tsv}\n')
        f.write(f'Output File : result/{OUTPUT_tsv}\n')
        # f.write(f'Pretrained Model : trained_model/model_TF_{INPUT_tsv_name}.zip\n')
        f.write(f'Train Data : {NUM_LINES["train"]}, Validation Data : {NUM_LINES["valid"]}, Test Data : {NUM_LINES["test"]}\n')
        f.write(f'Number of parameters : {params}\n')
        f.write(f'Training Time : {s2-s1}\n')
        f.write('\n')

    print('--- FINISH ---')
    print('INPUT :', INPUT_tsv)
    print(f'Train Data : {NUM_LINES["train"]}, Validation Data : {NUM_LINES["valid"]}, Test Data : {NUM_LINES["test"]}')
    print('Number of parameters :', params)
    print('Training Time :', s2-s1)

