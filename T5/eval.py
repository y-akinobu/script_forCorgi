import Levenshtein
from nltk import word_tokenize
from nltk import bleu_score
import pandas as pd
import csv
import datetime
import argparse

# オプション
parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('input_file', help='corpus')

args = parser.parse_args()

d_now = datetime.datetime.now()

PATH = args.input_file
PATH = PATH[PATH.rfind('/')+1:]
PATH_name = PATH.replace('.tsv', '')

df = pd.read_csv(args.input_file, sep='\t', names=('JPN', 'Py', 'Py_predict'))

def per_correct():
    '''
    コードの完全正解率
    '''

    cnt = 0
    correct = 0

    df_error = pd.DataFrame(columns=['JPN', 'Py', 'Py_predict'])

    for index, row in df.iterrows():
        py = row['Py'].strip()
        pred = row['Py_predict'].strip()

        cnt += 1

        if py == pred:
            correct += 1
        else:
            df_error = df_error.append({'JPN': row['JPN'], 'Py': py, 'Py_predict': pred}, ignore_index=True)

    df_error.to_csv(f'result/error/{PATH_name}_error.tsv', index=False, header=False, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, escapechar='\n')

    return correct / cnt * 100

def leven_bleu():
    '''
    レーベンシュタイン距離 (類似度算出; 1.0 - 編集距離)
    BLEU
    '''
    sum_l = 0
    sum_b = 0

    for index, row in df.iterrows():
        py = row['Py'].strip()
        pred = row['Py_predict'].strip()

        py = py.strip()
        pred = pred.strip()

        sum_l += Levenshtein.ratio(py, pred)
        sum_b += bleu_score.sentence_bleu(py.split(), pred.split())

    leven = sum_l / len(df)
    bleu = sum_b / len(df)

    return leven, bleu

if __name__ == '__main__':
    per_correct = per_correct()
    leven, bleu = leven_bleu()

    with open('log/evallog_mT5.txt', mode= 'a') as f:
        f.write(d_now.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write(f'Correct Rate : {per_correct}\n')
        f.write(f'Levenshtein Distance : {leven}\n')
        f.write(f'BLEU : {bleu}\n')
        f.write('\n')
