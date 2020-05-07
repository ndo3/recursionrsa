from collections import Counter
import pandas as pd
import string
from collections import namedtuple, defaultdict
import csv
import sys
import torch

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

    return df_final

# Literal listener data function

def get_literal_listener_training_data(df):
    return output # [correct_referent_idx, list_of_three_referents, descriptor]

# Literal Speaker data function

def get_literal_speaker_training_data(df): #TODO: Josh implement this
    output = []
    utterances = {}
    for row in df.iterrows():
        utt = row['contents']
        color = torch.tensor(row[['clickColH', 'clickColS', 'clickColL']])
        if utt not in utterances:
            utterances[utt] = len(utt)
        output.append([color, utterances[utt]])

    return output, utterances # [referent, utterance_idx], {utterance: idx forall utterances}