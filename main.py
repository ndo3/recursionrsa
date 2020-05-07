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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nlp = English()
tokenizer = Tokenizer(nlp.vocab)

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

def train_literals(listener_training_data, speaker_training_data, l0, s0, epochs):
    speaker_losses = []
    listener_losses = []
    listener_criterion = nn.CrossEntropyLoss()
    speaker_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([x for x in l0.parameters()] + [x for x in s0.parameters()], lr=5e-4)
    for i in range(epochs):
        print("Epoch", i)
        zipped_data = zip(listener_training_data, speaker_training_data)
        for (correct_referent_idx, list_of_three_referents, descriptor), (referent, utterance_idx) in tqdm(zipped_data, total=len(listener_training_data)):
            listener_probs = l0(referents=list_of_three_referents, descriptor=descriptor)
            listener_loss = listener_criterion(listener_probs.unsqueeze(0), correct_referent_idx.unsqueeze(0))
            listener_losses.append(listener_loss.item())

            speaker_probs = s0(referent)
            speaker_loss = speaker_criterion(speaker_probs.unsqueeze(0), utterance_idx.unsqueeze(0))
            speaker_losses.append(speaker_loss.item())

            loss = listener_loss + speaker_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return np.array(speaker_losses), np.array(listener_losses)


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
    referent_encoder = ReferentEncoder().to(device)
    description_encoder = DescriptionEncoder(vocab_size=len(label_encoder.classes_)).to(device)
    choice_ranker = ChoiceRanker("choiceranker").to(device)
    referent_describer = ReferentDescriber(num_utterances=len(label_encoder.classes_)).to(device)

    # Instantiate Literal Speaker and Literal Listener
    l0 = LiteralListener("literallistener", choice_ranker, referent_encoder, description_encoder).to(device)
    s0 = LiteralSpeaker("literalspeaker", referent_encoder, referent_describer).to(device)

    NUM_EPOCHS = 1
    smoothing_sigma = 2
    alpha = 0.5

    print("Training Literals")
    literal_speaker_losses, literal_listener_losses = train_literals(literal_listener_training_data, literal_speaker_training_data, l0, s0, NUM_EPOCHS)
    plt.plot(gaussian_filter1d(literal_listener_losses, sigma=smoothing_sigma), alpha=alpha)
    plt.plot(gaussian_filter1d(literal_speaker_losses, sigma=smoothing_sigma), alpha=alpha)

    plt.legend(["Literal Listener", "Literal Speaker"])
    plt.title("Literal Training Losses (Lower is Better)")
    plt.savefig("training_losses.png")

    s0.training = False
    l0.training = False


if __name__ == "__main__":
    main()
    # train_literal_listener(torch_all_colors, torch_all_descriptions, l0)
    # # train_literal_speaker()
    # for max_levels in range(5):
    #     speakers, listeners = run_reaconing(l0, s0, all_train_utterances, levels_of_recursion=max_levels)