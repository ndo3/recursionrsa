from models import LiteralListener, LiteralSpeaker, ReasoningListener, ReasoningSpeaker, ChoiceRanker, ReferentEncoder, DescriptionEncoder, ReferentDescriber
from scipy.special import logsumexp
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
# data loading
import pickle
import numpy as np
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from itertools import *
from sklearn import preprocessing
import sys
import pandas as pd
from gather_data import get_data, get_literal_listener_training_data, get_literal_speaker_training_data

nlp = English()
tokenizer = Tokenizer(nlp.vocab)


##############################
### HELPER FUNCTIONS #########
##############################

def getvocabsize(all_all_sentences):
    worddict = {}
    for all_sentences in all_all_sentences:
        for sentence in all_sentences:
            tokens = tokenizer(sentence)
            for token in tokens:
                worddict[token] = 1
    return len(worddict)


def getalldescriptors(all_all_sentences):
    sentence_dict = {}
    for all_sentences in all_all_sentences:
        for sentence in all_sentences:
            if sentence not in sentence_dict: sentence_dict[sentence] = 1
            else: sentence_dict[sentence] += 1
    filtered = {s: sentence_dict[s] for s in sentence_dict if sentence_dict[s] >= 2}
    return list(filtered.keys())


def getallreferents(ref_index):
    print("OMANGUA ", list(ref_index)[0])
    ref_list = list(ref_index)
    ref_list = [r[1:-1].split(", ") for r in ref_list]
    ref_list = [[int(r[0]), int(r[1]), int(r[2])] for r in ref_list]
    return np.array(ref_list)


#########################################################################################


def train_literal_listener(training_data, model, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    for i in range(epochs):
        for correct_referent_idx, list_of_three_referents, descriptor in training_data:
            probs = model.forward(referents=list_of_three_referents, descriptor=descriptors)

            # calculate cross entropy loss
            loss = criterion(probs, correct_referent_idx)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_literal_speaker(training_data, model, epochs):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    for i in range(epochs):
        for referent, utterance_idx in training_data:
            probs = model.forward(referent)

            loss = criterion(probs, utterance_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_reasoning(literal_listener, literal_speaker, utterances, levels_of_recursion=5):
    speakers = {}
    listeners = {}
    for i in range(levels_of_recursion):
        if i == 0:
            listeners[0] = l0
            speakers[0] = s0
            continue # because i == 0 will mean that we're at literal speaker and listener level
        else:
            listeners[i] = ReasoningListener("l"+str(i), speakers[i-1])
            speakers[i] = ReasoningSpeaker("s"+str(i), listeners[i], s0, utterances)
    return speakers, listeners


def main():
    # LOAD DATA
    data_df = get_data()
    literal_speaker_training_data, utterances_dict = get_literal_speaker_training_data(data_df)
    literal_listener_training_data = get_literal_listener_training_data(data_df)


    # Instantiate Modules
    referent_encoder = ReferentEncoder()
    description_encoder = DescriptionEncoder(vocab_size=len(utterances_dict))
    choice_ranker = ChoiceRanker("choiceranker")
    referent_describer = ReferentDescriber(num_utterances=len(utterances_dict))

    # Instantiate Literal Speaker and Literal Listener
    l0 = LiteralListener("literallistener", choice_ranker, referent_encoder, description_encoder)
    s0 = LiteralSpeaker("literalspeaker", referent_encoder, referent_describer)

    # Data
    NUM_EPOCHS = 10

    train_literal_listener(literal_listener_training_data, l0, epochs=NUM_EPOCHS)
    train_literal_speaker(literal_speaker_training_data, s0, epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()
    # train_literal_listener(torch_all_colors, torch_all_descriptions, l0)
    # # train_literal_speaker()
    # for max_levels in range(5):
    #     speakers, listeners = run_reaconing(l0, s0, all_train_utterances, levels_of_recursion=max_levels)