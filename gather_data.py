from collections import Counter
import pandas as pd
import string
from collections import namedtuple, defaultdict
import csv
import sys
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def get_data():
    df = pd.read_csv("./data/filteredCorpus.csv")
    df_filt = df[df['outcome']==True] # use only successful games
    df_filt = df_filt[df_filt['role']=='speaker'] # use speaker utterances
    df_filt = df_filt[df_filt['source']=='human'] # use speaker utterances

    # making a list of utterances that we want to use, so we can take these rows from df_filt
    utt = df_filt['contents']
    utt_filt = [u.lower() for u in utt if len(u.split()) == 1] # only use one word utterances
    utt_filt = [u.translate(str.maketrans('', '', string.punctuation)) for u in utt_filt] # remove punctuation
    utt_final = list((Counter(utt_filt) - Counter(set(utt_filt))).keys()) # use utterances that appear more than once

    # df_filt = df_filt[df_filt['numCleanWords'] == 1]
    df_filt['contents'] = df_filt['contents'].apply(lambda x: x.lower())
    df_filt['contents'] = df_filt['contents'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))# filter to take out punctuation
    df_final = df.loc[df['contents'].isin(utt_final)] # this is the dataset of all the games that we want to use

    le = LabelEncoder()
    df_final['contents'] = le.fit_transform(df_final['contents'])

    return df_final, le



def get_meaning_matrix(df):
    df['colors'] = list(zip(df['clickColH'], df['clickColS'], df['clickColL']))
    df['colors'] = df['colors'].apply(lambda x: str(x))
    colors_le = LabelEncoder()
    df['colors'] = colors_le.fit_transform(df['colors']) # 100 x 100 (test data)
    print("length colors and contents", len(df['colors']), len(df['contents']))
    print("set colors and contents", len(set(df['colors'])), len(set(df['contents'])))
    meaning_mat = pd.crosstab(df['colors'], df['contents']) # rows are colors, columns are utterances
    # row numbers and column numbers correspond to labels from colors_le and le (utterances) from get_data()
    meaning_mat = np.array(meaning_mat) # a num_color x num_utterances matrix

    for i in range(len(meaning_mat[:,0])):
        if sum(meaning_mat[i,:]) == 0:
            print("meaning mat is 0 for this row: ", i)
        for j in range(len(meaning_mat[0,:])):
            if meaning_mat[i,j] == 0:
                print("meaning mat is 0 at: ", i,j," !!!")
    return meaning_mat, colors_le




# Literal listener data function




def get_pragmatic_listener_testing_data(df):
    output = []
    all_utt = list(set(list(df['contents'])))
    desc_to_idx = {u: i for i,u in enumerate(all_utt)}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        utt = torch.tensor(row['contents']).to(device)
        correct = torch.tensor(row[['clickColH', 'clickColS', 'clickColL']], dtype=torch.float32)
        alt1 = torch.tensor(row[['alt1ColH', 'alt1ColS', 'alt1ColL']], dtype=torch.float32)
        alt2 = torch.tensor(row[['alt2ColH', 'alt2ColS', 'alt2ColL']], dtype=torch.float32)
        colors = (correct, alt1, alt2)
        # idxs = random.choice([0,1,2]) # randomly permute colors
        idxs = np.arange(3)
        np.random.shuffle(idxs)
        colors_shuff = torch.stack([colors[idxs[0]], colors[idxs[1]], colors[idxs[2]]]).to(device)
        correct_idx = torch.tensor(idxs[0], dtype=torch.long).to(device) # index where correct color goes
        output.append((correct_idx, colors_shuff, utt))
    return output, all_utt, desc_to_idx # [correct_referent_idx, list_of_three_referents, descriptor_idx] desc_to_idx idx_to_desc

    # return all_utt, idx_to_desc # [correct_referent_idx, list_of_three_referents, descriptor_idx] desc_to_idx idx_to_desc




def get_literal_listener_training_data(df):
    output = []
    all_utt = df['contents']
    idx_to_desc = {i: u for i,u in enumerate(all_utt)}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        utt = torch.tensor(row['contents']).to(device)
        correct = torch.tensor(row[['clickColH', 'clickColS', 'clickColL']], dtype=torch.float32)
        alt1 = torch.tensor(row[['alt1ColH', 'alt1ColS', 'alt1ColL']], dtype=torch.float32)
        alt2 = torch.tensor(row[['alt2ColH', 'alt2ColS', 'alt2ColL']], dtype=torch.float32)
        colors = (correct, alt1, alt2)
        # idxs = random.choice([0,1,2]) # randomly permute colors
        idxs = np.arange(3)
        np.random.shuffle(idxs)
        colors_shuff = torch.stack([colors[idxs[0]], colors[idxs[1]], colors[idxs[2]]]).to(device)
        correct_idx = torch.tensor(idxs[0], dtype=torch.long).to(device) # index where correct color goes
        output.append((correct_idx, colors_shuff, utt))
    return output#, all_utt, idx_to_desc # [correct_referent_idx, list_of_three_referents, descriptor_idx] desc_to_idx idx_to_desc

# Literal Speaker data function - hi r u ok

def get_literal_speaker_training_data(df):
    output = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        utt = torch.tensor(row['contents'], dtype=torch.long).to(device)
        color = torch.tensor(row[['clickColH', 'clickColS', 'clickColL']], dtype=torch.float32).to(device)
        output.append([color, utt])

    return output # [referent, utterance_idx]
