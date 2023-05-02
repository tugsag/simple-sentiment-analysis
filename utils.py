import glob
import json
import os
import nltk
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


#TODO: extend length of seqs under 10, remove lemmatization and see what happens

def get_data():
    # Just get all data and stratify later
    dfs = []
    for p in glob.glob('data/*.txt'):
        dfs.append(pd.read_csv(p, sep=';', header=None))

    df = pd.concat(dfs)
    df.rename(columns={1: 'label', 0: 'text'}, inplace=True)
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['label'])

    write_path = 'data/encode2label.json'
    if not os.path.exists(write_path):
        print(f'Writing invkey to {write_path}')
        invkey = {int(v):k for k, v in zip(le.classes_, le.transform(le.classes_))}
        with open(write_path, 'w') as f:
            json.dump(invkey, f)
    return df

def preprocess_data(x):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(x.lower())

    sw = set(nltk.corpus.stopwords.words('english'))
    lemma = nltk.stem.WordNetLemmatizer()
    filtered = [lemma.lemmatize(word, pos='v') for word in tokenized if word not in sw]

    return ' '.join(filtered)

def build_vocab(df):
    allstring = ' '.join(df['text'].to_list())
    uniques = np.unique(allstring.split(' '))
    map = {w:i+1 for i, w in enumerate(uniques)}
    # maybe have a special int for oov 
    map['<PAD>'] = 0
    map['<UNK>'] = uniques.shape[0]+1
    return map


def get_and_preprocess(preprocess=True):
    df = get_data()
    if preprocess:
        df['text'] = df['text'].apply(preprocess_data)
    vocab = build_vocab(df)

    write_path = 'data/vocab.json'
    if not os.path.exists(write_path):
        print(f'Writing vocab to {write_path}')
        with open(write_path, 'w') as f:
            json.dump(vocab, f)
    return df, vocab