import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import zipfile
import argparse

# オプション
parser = argparse.ArgumentParser(description='Using mT5 with ABCI')
parser.add_argument('zip_file', nargs='+', help='corpus')  

args = parser.parse_args()

with zipfile.ZipFile(args.zip_file) as existing_zip:
    existing_zip.extractall('data/temp/ext')

MODEL_DIR = "/content/model"

# トークナイザー（SentencePiece）
tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True)

# 学習済みモデル
trained_model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)

# GPUの利用有無
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    trained_model.cuda()

def translate(text):
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    if USE_GPU:
        input_ids = input_ids.cuda()
    predict = trained_model.generate(input_ids)
    return tokenizer.decode(predict[0], skip_special_tokens=True)

def translate_beam(text: str, beams=5, max_length=64):
        trained_model.config.update({"num_beams": beams})
        input_ids = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            # padding="max_length",
            padding="do_not_pad",
            truncation=True,
            return_tensors='pt').input_ids

        if USE_GPU:
            input_ids = input_ids.cuda()
        predict = trained_model.generate(input_ids,
                                          return_dict_in_generate=True,
                                          output_scores=True,
                                          length_penalty=8.0,
                                          max_length=max_length,
                                          num_return_sequences=beams,
                                          early_stopping=True)
        pred_list = sorted([[tokenizer.decode(predict.sequences[i], skip_special_tokens=True),
                             predict.sequences_scores[i].item()] for i in range(len(predict))], key=lambda x: x[1], reverse=True)
        sentences_list = [i[0] for i in pred_list]
        scores_list = [i[1] for i in pred_list]
        # return sentences_list, scores_list
        return sentences_list[0]

if __name__ == '__main__':
    text = [
        '<A> の先頭を表示する'
    ]
    for t in text:
        predict = translate_beam(t)
        print(predict)
