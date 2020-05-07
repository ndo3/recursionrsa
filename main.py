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
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

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
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    for i in range(epochs):
        print("Epoch", i)
        for correct_referent_idx, list_of_three_referents, descriptor in tqdm(training_data):
            probs = model(referents=list_of_three_referents, descriptor=descriptor)

            # calculate cross entropy loss
            loss = criterion(probs.unsqueeze(0), correct_referent_idx.unsqueeze(0))
            losses.append(loss.item())

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return np.array(losses)


def train_literal_speaker(training_data, model, epochs):
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    for i in range(epochs):
        print("Epoch", i)
        for referent, utterance_idx in tqdm(training_data):
            probs = model(referent)

            loss = criterion(probs.unsqueeze(0), utterance_idx.unsqueeze(0))
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return np.array(losses)


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
    print("Loading Data")
    data_df, label_encoder = get_data()
    # data_df = data_df.head()
    literal_speaker_training_data = get_literal_speaker_training_data(data_df)
    literal_listener_training_data = get_literal_listener_training_data(data_df)


    print("Instantiating Models")
    # Instantiate Modules
    referent_encoder = ReferentEncoder()
    description_encoder = DescriptionEncoder(vocab_size=len(label_encoder.classes_))
    choice_ranker = ChoiceRanker("choiceranker")
    referent_describer = ReferentDescriber(num_utterances=len(label_encoder.classes_))

    # Instantiate Literal Speaker and Literal Listener
    l0 = LiteralListener("literallistener", choice_ranker, referent_encoder, description_encoder)
    s0 = LiteralSpeaker("literalspeaker", referent_encoder, referent_describer)

    NUM_EPOCHS = 1
    smoothing_sigma = 2
    alpha = 0.5
    print("Training Literal Litener")
    literal_listener_losses = train_literal_listener(literal_listener_training_data, l0, epochs=NUM_EPOCHS)
    plt.plot(gaussian_filter1d(literal_listener_losses, sigma=smoothing_sigma), alpha=alpha)
    print("Training Literal Speaker")
    literal_speaker_losses = train_literal_speaker(literal_speaker_training_data, s0, epochs=NUM_EPOCHS)
    plt.plot(gaussian_filter1d(literal_speaker_losses, sigma=smoothing_sigma), alpha=alpha)
    plt.legend(["Literal Listener", "Literal Speaker"])
    plt.show()

    s0.training = False
    l0.training = False


if __name__ == "__main__":
    main()
    # train_literal_listener(torch_all_colors, torch_all_descriptions, l0)
    # # train_literal_speaker()
    # for max_levels in range(5):
    #     speakers, listeners = run_reaconing(l0, s0, all_train_utterances, levels_of_recursion=max_levels)