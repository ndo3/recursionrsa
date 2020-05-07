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

TRAIN = 0.8
# load the data from the colors_probs.pkl file
with open("data/colors_probs_better.pkl", "rb") as datafile:
    data = pickle.load(datafile)
    # print(data)
    indices = np.random.rand(len(data)) < TRAIN
    traindata, testdata = data[indices].sample(frac=1), data[~indices].sample(frac=1)


allwordsdict = {}
# this is just for all the training data
all_train_utterances = traindata['utterances']
# this variable is for all the possible utterances - so we can see the entire distribution
all_utterances = list(getalldescriptors(data['utterances']))
# print(all_utterances)
all_colors = getallreferents(traindata.index.values).astype(np.float32)
# print(all_colors.dtype)
# sys.exit()
# encoding all the description
le = preprocessing.LabelEncoder() #LabelEncoder for all the possible utterances
transformed_utterances = le.fit_transform(all_utterances)



# # instantiate referent and description encoder
referent_encoder = ReferentEncoder()
description_encoder = DescriptionEncoder(vocab_size=getvocabsize(data['utterances']))
choice_ranker = ChoiceRanker("choiceranker", referent_encoder, description_encoder, num_descriptors=len(getalldescriptors(all_train_utterances)), num_referents=len(traindata))
referent_describer = ReferentDescriber(num_utterances=len(getalldescriptors(all_train_utterances)))
# # instantiate the LiteralSpeaker and the LiteralListener
l0 = LiteralListener("literallistener", choice_ranker)
s0 = LiteralSpeaker("literalspeaker", referent_encoder, referent_describer)

# # we're going to set the optimizer to be based on cross entropy loss
criterion = nn.CrossEntropyLoss()
# optimizer for literal listener
l0_parameters = chain(choice_ranker.parameters(), referent_encoder.parameters(), description_encoder.parameters())
optimizer_literal_listener = optim.SGD(l0_parameters, lr=0.001, momentum=0.9)
# optimizer for literal speaker
s0_parameters = chain(referent_encoder.parameters(), referent_describer.parameters())
optimizer_literal_speaker = optim.SGD(s0_parameters, lr=0.001, momentum=0.9)

# calculate the encoded referents and descriptions stuff here
# convert to torch
torch_all_colors = torch.from_numpy(all_colors)
torch_all_descriptions = torch.from_numpy(transformed_utterances)

# # Data
NUM_EPOCHS = 10


# THE OTHER DATA
# {utterance -> [colors]}
# with open("utterances_prob_better.pkl", "rb") as utterances_
literal_training_data = pd.read_pickle("utterances_prob_better.pkl")
literal_training_data = {le.transform([k])[0]: literal_training_data[k] for k in literal_training_data if k in all_utterances}




def train_literal_listener(training_data, model, epochs=NUM_EPOCHS):
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


    # for i in range(epochs):
    #     for d_idx in range(len(torchified_descriptors)):
    #         description = torchified_descriptors[d_idx]
    #         e_referents = referent_encoder.forward(torchified_referents)
    #         e_descriptors = description_encoder.forward(torchified_descriptors)
    #         # zero the parameters dimension
    #         optimizer_literal_listener.zero_grad()
    #         # clarify the dimensions here
    #         probs = model.forward(encoded_referents=e_referents, descriptor_idx=d_idx, encoded_descriptors=e_descriptors)
    #         print(probs.shape)
    #         sys.exit()
    #         # calculate the cross entropy loss
    #         loss = criterion(probs, labels)
    #         # update
    #         loss.backward()
    #         optimizer_literal_listener.step()


def train_literal_speaker(all_referents, all_descriptors, model, epochs=NUM_EPOCHS):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    for i in range(epochs):
        for referent, utterance_idx in training_data:
            probs = model.forward(referent)

            loss = criterion(probs, utterance_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        # for r_idx, referent in enumerate(all_referents):
        #     # referent = referent.unsqueeze(0)
        #     # e_referents = [referent_encoder.forward(x) for x in torch_all_colors]
        #     e_descriptors = [description_encoder.forward(x) for x in torch_all_descriptions]
        #     # zero the parameters dimension
        #     optimizer_literal_speaker.zero_grad()
        #     probs = model.forward(referent=referent, correct_choice=1, utterances=e_descriptors)
        #     # calculate cross entropy loss
        #     loss = criterion(probs, labels)
        #     # update
        #     loss.backward()
        #     optimizer_literal_speaker.step()


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



if __name__ == "__main__":
    train_literal_listener(torch_all_colors, torch_all_descriptions, l0)
    # train_literal_speaker()
    for max_levels in range(5):
        speakers, listeners = run_reaconing(l0, s0, all_train_utterances, levels_of_recursion=max_levels)