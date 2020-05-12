from models import LiteralListener, LiteralSpeaker, ReasoningListener, ReasoningSpeaker, ChoiceRanker, ReferentEncoder, DescriptionEncoder, ReferentDescriber, ClassicLiteralListener, ClassicLiteralSpeaker
from scipy.special import logsumexp
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from gather_data import get_data, get_literal_listener_training_data, get_literal_speaker_training_data, get_pragmatic_listener_testing_data, get_meaning_matrix
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
import sys
import random
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy as cp
from random import choices
import json

np.random.seed(0)
random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
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
    for i in range(epochs):
        print("Epoch", i)
        np.random.shuffle(zipped_data)
        for (correct_referent_idx, list_of_three_referents, descriptor), (referent, utterance_idx) in tqdm(zipped_data):
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


def create_reasoning_entities(literal_listener, literal_speaker, utterances, speaker_alpha, levels_of_recursion):
    speakers = {}
    listeners = {}
    for i in range(0, levels_of_recursion+1):
        if i == 0:
            listeners[0] = literal_listener
            speakers[0] = literal_speaker
        else:
            listeners[i] = ReasoningListener("l"+str(i), speakers[i-1])
            speakers[i] = ReasoningSpeaker("s"+str(i), listeners[i-1], speakers[0], utterances, speaker_alpha)
    return speakers, listeners


def run_reasoning(pragmatic_listener_testing_data, literal_listener, literal_speaker, all_utterances, d_idx_to_descriptor, alpha):
    """
    Assumption: pragmatic listener testing data:
    """
    reasoning_data = 0
    result_dict = {}
    dynamic_dict = {}
    print("Num utterances: ", len(all_utterances))
    for i in range(0, 12, 2):
        print("Max level: ", i)
        speakers, listeners = create_reasoning_entities(literal_listener, literal_speaker, all_utterances, alpha, levels_of_recursion=i)
        # print(speakers, listeners)
        num_correct = 0.
        for correct_referent_idx, list_of_three_referents, descriptor_idx in tqdm(pragmatic_listener_testing_data):
            # grab the last listener distribution
            if i == 0:
                prob_dist = listeners[i](referents=list_of_three_referents, descriptor=torch.tensor(d_idx_to_descriptor[int(descriptor_idx)], device=device), descriptor_idx=descriptor_idx, descriptors=all_utterances)
            else:
                prob_dist = listeners[i](referents=list_of_three_referents, descriptor=torch.tensor(d_idx_to_descriptor[int(descriptor_idx)], device=device), descriptor_idx=descriptor_idx, descriptors=all_utterances, dynamic_dict=dynamic_dict) #TODO: Change the descriptor_idx value


            r_idx = torch.argmax(prob_dist)
            if r_idx == correct_referent_idx:
                num_correct += 1.
        result_dict[i] = num_correct / len(pragmatic_listener_testing_data)
    print(result_dict)
    return result_dict


def helper__convert_to_string_color(inp):
    inp = [int(ele) for ele in list(inp.numpy())]
    ret = "("
    for ele in inp:
        ret += str(ele) + ', '
    ret = list(ret)[:len(ret)-1]
    ret[-1] = ")"
    return "".join(ret)


def run_classic(df, pragmatic_listener_testing_data, alpha):
    classical_listener = ClassicLiteralListener("classicall0")
    classical_speaker = ClassicLiteralSpeaker("s0classic", alpha)
    meaning_mat, colors_le = get_meaning_matrix(df)
    print("shape: ", meaning_mat.shape)
    result_dict = {i: 0 for i in range(0, 12, 2)}

    # for loop over referents and descriptors:
    for correct_referent_idx, list_of_three_referents, descriptor_idx in tqdm(pragmatic_listener_testing_data):
        actual_list_of_three_referents = [helper__convert_to_string_color(x) for x in list_of_three_referents]
        
        # SM = three rows of meaning mat 
        list_of_ids_of_three_actual_referents = [-1, -1, -1]
        for i, color in enumerate(actual_list_of_three_referents):
            if color in colors_le.classes_:
                list_of_ids_of_three_actual_referents[i] = colors_le.transform([color])[0]
                # else continue
                # so, if a color doesn't exist in the label encoder, it has label -1
        encoded_colors = list_of_ids_of_three_actual_referents
        # [45, 70] .forward([45, 70])
        SM = [[1/348]*348, [1/348]*348, [1/348]*348]
        for i, color in enumerate(encoded_colors):
            SM[i] = meaning_mat[color]

        SM = np.array(SM)

        for num_rec in range(0, 12, 2):
            SM = classical_listener.forward(SM)
            lol = list(cp(SM))
            SM = classical_speaker.forward(SM)
            # print(sum(sum([l == s for l,s in zip(lol, list(SM))])))
            correct = correct_referent_idx == np.argmax(softmax(alpha * np.apply_along_axis(softmax, 0, SM)[:,descriptor_idx]))
            #if correct: 
            result_dict[num_rec] += softmax(alpha * np.apply_along_axis(softmax, 0, SM)[:,descriptor_idx])[correct_referent_idx]
            num_rec += 2
    result_dict = {k: v/len(pragmatic_listener_testing_data) for k,v in result_dict.items()}
    return result_dict

        

def plot_reasoning_data(level_to_accuracy):
    xs = sorted([k for k in level_to_accuracy])
    ys = [level_to_accuracy[k] for k in xs]
    plt.plot(xs, ys)
    plt.title("Accuracy vs Number of Recursions")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Recursions")
    plt.savefig("accuracy_vs_recursions.png")


def calculate_accuracy(data, listener, speaker):
    num_correct = 0.
    for correct_referent_idx, list_of_three_referents, _ in data:
        descriptor = torch.argmax(speaker(list_of_three_referents[correct_referent_idx])) # Descriptor chooses an utterance
        guess = torch.argmax(listener(list_of_three_referents, descriptor)) # Listener makes a guess as to the referent

        if guess == correct_referent_idx:
            num_correct += 1.

    return num_correct / len(data)


def write_results(alpha, result_dict, output_file):
    if output_file:
        with open(output_file, "a") as f:
            f.write(str(alpha))
            f.write("|")
            f.write(str(result_dict))
            f.write("\n")



def main(training=True, alpha = 1, output_file = None):
    # LOAD DATA
    print("Loading Data")
    data_df, label_encoder = get_data()
    encoded_distinct_utterances = label_encoder.transform(label_encoder.classes_)
    # data_df = data_df.head()
    # data_df = data_df[:500]

    label_encoder_classic = LabelEncoder()
    label_encoder_classic.fit(data_df['contents'])
    data_df_classic = data_df
    data_df_classic['contents'] = data_df_classic['contents'].apply(lambda x: label_encoder_classic.transform([x])[0])

    training_split = 0.8
    training_df = data_df[:int(training_split * len(data_df))]
    testing_df = data_df[int(training_split * len(data_df)):]
    print(len(data_df), len(training_df), len(testing_df))
    literal_speaker_training_data = get_literal_speaker_training_data(training_df)
    literal_listener_training_data  = get_literal_listener_training_data(training_df)

    literal_speaker_testing_data = get_literal_speaker_training_data(testing_df)
    # pragmatic_listener_testing_data, descriptors, test_idx_to_desc = get_pragmatic_listener_testing_data(testing_df)
    pragmatic_listener_testing_data, descriptors, test_idx_to_desc = get_pragmatic_listener_testing_data(testing_df)



    print("Instantiating Models")
    # Instantiate Modules
    referent_encoder = ReferentEncoder().to(device)
    description_encoder = DescriptionEncoder(vocab_size=len(label_encoder.classes_)).to(device)
    choice_ranker = ChoiceRanker("choiceranker").to(device)
    referent_describer = ReferentDescriber(num_utterances=len(label_encoder.classes_)).to(device)

    # Instantiate Literal Speaker and Literal Listener
    l0 = LiteralListener("literallistener", choice_ranker, referent_encoder, description_encoder).to(device)
    s0 = LiteralSpeaker("literalspeaker", referent_encoder, referent_describer, alpha=alpha).to(device)

    NUM_EPOCHS = 20
    smoothing_sigma = 2
    clip_bound = 20.

    if training:
        print("Training Literals")
        literal_speaker_losses, literal_listener_losses = train_literals(literal_listener_training_data, literal_speaker_training_data, l0, s0, NUM_EPOCHS)

        torch.save(s0.state_dict(), "literal_speaker_alpha_" + str(alpha) + "_datapoints" + str(len(data_df)) + ".pth")
        torch.save(l0.state_dict(), "literal_listener_alpha_" + str(alpha) + "_datapoints" + str(len(data_df)) + ".pth")

        literal_listener_losses = np.clip(gaussian_filter1d(literal_listener_losses, sigma=smoothing_sigma), 0., clip_bound)
        literal_speaker_losses = np.clip(gaussian_filter1d(literal_speaker_losses, sigma=smoothing_sigma), 0., clip_bound)
        losses_df = pd.DataFrame({"Literal Speaker": literal_speaker_losses, "Literal Listener": literal_listener_losses})
        losses_df["Number of Examples"] = pd.Series(np.arange(len(losses_df)))
        losses_df = losses_df.melt(value_vars=["Literal Speaker", "Literal Listener"], id_vars="Number of Examples")
        losses_df.columns = ["Number of Examples", "Type", "Loss"]

        sns.lineplot(y="Loss", x="Number of Examples", hue="Type", data=losses_df)
        plt.savefig("losses.png")
        sys.exit()
    else:
        print("Loading Previously Saved Literal Weights")
        if device == 'cpu':
            s0.load_state_dict(torch.load("literal_speaker_alpha_" + str(alpha) + "_datapoints" + str(len(data_df)) + ".pth", map_location=torch.device('cpu')))
            l0.load_state_dict(torch.load("literal_listener_alpha_" + str(alpha) + "_datapoints" + str(len(data_df)) + ".pth", map_location=torch.device('cpu')))
        else:
            s0.load_state_dict(torch.load("literal_speaker_alpha_" + str(alpha) + "_datapoints" + str(len(data_df)) + ".pth"))
            l0.load_state_dict(torch.load("literal_listener_alpha_" + str(alpha) + "_datapoints" + str(len(data_df)) + ".pth"))

    s0.training = False
    l0.training = False

    # training_accuracy = calculate_accuracy(literal_listener_training_data, l0, s0)
    # testing_dataset = get_literal_listener_training_data(testing_df)
    # testing_accuracy = calculate_accuracy(testing_dataset, l0, s0)

    # print("Training Accuracy", training_accuracy, "Testing Accuracy", testing_accuracy)
    # print(len(test_idx_to_desc), len(descriptors))
    result_dict = run_reasoning(pragmatic_listener_testing_data, l0, s0, torch.tensor(encoded_distinct_utterances, device=device), {i: u for i, u in enumerate(encoded_distinct_utterances)}, alpha)
    print("finished calculating results!!")
    write_results(alpha, result_dict, output_file)
    plot_reasoning_data(result_dict)

    ##### Nam's part for classical

    testing_df_classic = data_df_classic[int(training_split * len(data_df_classic)):]
    pragmatic_listener_testing_data_classic, _, _ = get_pragmatic_listener_testing_data(testing_df_classic)
    
    classical_results = run_classic(data_df_classic, pragmatic_listener_testing_data, alpha)
    with open(str(alpha) + "_classical.json", "w") as dumpfilehehe:
        json.dump(classical_results, dumpfilehehe)
    
    print("Finished dumping stuff!!!")

    # print(results)



if __name__ == "__main__":
    # Train for alpha = 0.25, 0.5, 0.75, 1., 1.5
    alpha = float(sys.argv[1])
    output_file = sys.argv[2]
    main(training=False, alpha = alpha, output_file = output_file)
