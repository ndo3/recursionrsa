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
from gather_data import get_data, get_literal_listener_training_data, get_literal_speaker_training_data, get_pragmatic_listener_testing_data
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
sns.set_style("darkgrid")

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
    zipped_data = [x for x in zip(listener_training_data, speaker_training_data)]
    for i in trange(epochs):
        # print("Epoch", i)
        np.random.shuffle(zipped_data)
        for (correct_referent_idx, list_of_three_referents, descriptor), (referent, utterance_idx) in zipped_data:
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


def create_reasoning_entities(literal_listener, literal_speaker, utterances, levels_of_recursion):
    speakers = {}
    listeners = {}
    for i in range(0, levels_of_recursion+1):
        if i == 0:
            listeners[0] = literal_listener
            speakers[0] = literal_speaker
        else:
            listeners[i] = ReasoningListener("l"+str(i), speakers[i-1])
            speakers[i] = ReasoningSpeaker("s"+str(i), listeners[i-1], speakers[0], utterances)
    return speakers, listeners


def run_reasoning(pragmatic_listener_testing_data, literal_listener, literal_speaker, all_utterances, d_idx_to_descriptor):
    """
    Assumption: pragmatic listener testing data: 
    """
    reasoning_data = 0
    result_dict = {}
    dynamic_dict = {}
    print("Num utterances: ", len(all_utterances))
    for i in range(0, 12, 2):
        print("Max level: ", i)
        speakers, listeners = create_reasoning_entities(literal_listener, literal_speaker, all_utterances, levels_of_recursion=i)
        # print(speakers, listeners)
        num_correct = 0.
        num_all = 0.
        for correct_referent_idx, list_of_three_referents, descriptor_idx in tqdm(pragmatic_listener_testing_data):
            # grab the last listener distribution
            if i == 0:
                prob_dist = listeners[i](referents=list_of_three_referents, descriptor=torch.tensor(d_idx_to_descriptor[int(descriptor_idx)], device=device), descriptor_idx=descriptor_idx, descriptors=all_utterances)
            else:
                prob_dist = listeners[i](referents=list_of_three_referents, descriptor=torch.tensor(d_idx_to_descriptor[int(descriptor_idx)], device=device), descriptor_idx=descriptor_idx, descriptors=all_utterances, dynamic_dict=dynamic_dict) #TODO: Change the descriptor_idx value


            r_idx = torch.argmax(prob_dist)
            if r_idx == correct_referent_idx:
                num_correct += 1.
            num_all += 1.
            # print(num_all, i)
            # if num_all > 10 and i > 0:
            #     print("\n\n\n\n\n\n\n\n\n\n")
            #     sys.exit()
        result_dict[i] = num_correct / len(pragmatic_listener_testing_data)
    print(result_dict)
    return result_dict


def plot_reasoning_data(level_to_accuracy):
    xs = sorted([k for k in level_to_accuracy])
    ys = [level_to_accuracy[k] for k in xs]
    plt.plot(xs, ys)
    plt.show()


def calculate_accuracy(data, listener, speaker):
    num_correct = 0.
    for correct_referent_idx, list_of_three_referents, _ in data:
        descriptor = torch.argmax(speaker(list_of_three_referents[correct_referent_idx])) # Descriptor chooses an utterance
        guess = torch.argmax(listener(list_of_three_referents, descriptor)) # Listener makes a guess as to the referent

        if guess == correct_referent_idx:
            num_correct += 1.

    return num_correct / len(data)


def main(training=True):
    # LOAD DATA
    print("Loading Data")
    data_df, label_encoder = get_data()
    encoded_distinct_utterances = label_encoder.transform(label_encoder.classes_)
    # data_df = data_df.head()
    # data_df = data_df[:500]
    training_split = 0.8
    training_df = data_df[:int(training_split * len(data_df))]
    testing_df = data_df[int(training_split * len(data_df)):]
    literal_speaker_training_data = get_literal_speaker_training_data(training_df)
    literal_listener_training_data  = get_literal_listener_training_data(training_df)

    literal_speaker_testing_data = get_literal_speaker_training_data(testing_df)
    pragmatic_listener_testing_data, descriptors, test_idx_to_desc = get_pragmatic_listener_testing_data(testing_df)



    print("Instantiating Models")
    # Instantiate Modules
    referent_encoder = ReferentEncoder().to(device)
    description_encoder = DescriptionEncoder(vocab_size=len(label_encoder.classes_)).to(device)
    choice_ranker = ChoiceRanker("choiceranker").to(device)
    referent_describer = ReferentDescriber(num_utterances=len(label_encoder.classes_)).to(device)

    # Instantiate Literal Speaker and Literal Listener
    l0 = LiteralListener("literallistener", choice_ranker, referent_encoder, description_encoder).to(device)
    s0 = LiteralSpeaker("literalspeaker", referent_encoder, referent_describer).to(device)

    NUM_EPOCHS = 100
    smoothing_sigma = 2
    alpha = 0.5
    clip_bound = 20.

    if training:
        print("Training Literals")
        literal_speaker_losses, literal_listener_losses = train_literals(literal_listener_training_data, literal_speaker_training_data, l0, s0, NUM_EPOCHS)

        torch.save(s0.state_dict(), "literal_speaker.pth")
        torch.save(l0.state_dict(), "literal_listener.pth")

        literal_listener_losses = np.clip(gaussian_filter1d(literal_listener_losses, sigma=smoothing_sigma), 0., clip_bound)
        literal_speaker_losses = np.clip(gaussian_filter1d(literal_speaker_losses, sigma=smoothing_sigma), 0., clip_bound)
        losses_df = pd.DataFrame({"Literal Speaker": literal_speaker_losses, "Literal Listener": literal_listener_losses})
        losses_df["Number of Examples"] = pd.Series(np.arange(len(losses_df)))
        losses_df = losses_df.melt(value_vars=["Literal Speaker", "Literal Listener"], id_vars="Number of Examples")
        losses_df.columns = ["Number of Examples", "Type", "Loss"]

        sns.lineplot(y="Loss", x="Number of Examples", hue="Type", data=losses_df)
        plt.savefig("losses.png")
    else:
        print("Loading Previously Saved Literal Weights")
        if device == 'cpu':
            s0.load_state_dict(torch.load("literal_speaker.pth", map_location=torch.device('cpu')))
            l0.load_state_dict(torch.load("literal_listener.pth", map_location=torch.device('cpu')))
        else:
            s0.load_state_dict(torch.load("literal_speaker.pth"))
            l0.load_state_dict(torch.load("literal_listener.pth"))

    s0.training = False
    l0.training = False

    training_accuracy = calculate_accuracy(literal_listener_training_data, l0, s0)
    testing_dataset = get_literal_listener_training_data(testing_df)
    testing_accuracy = calculate_accuracy(testing_dataset, l0, s0)

    print("Training Accuracy", training_accuracy, "Testing Accuracy", testing_accuracy)
    print(len(test_idx_to_desc), len(descriptors))
    result_dict = run_reasoning(pragmatic_listener_testing_data, l0, s0, torch.tensor(encoded_distinct_utterances, device=device), {i: u for i, u in enumerate(encoded_distinct_utterances)})
    plot_reasoning_data(result_dict)


if __name__ == "__main__":
    main(training=False)
