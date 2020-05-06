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


#########################################################################################

TRAIN = 0.8
# load the data from the colors_probs.pkl file
with open("data/colors_probs.pkl", "rb") as datafile:
    data = pickle.load(datafile)
    # print(data)
    indices = np.random.rand(len(data)) < TRAIN
    traindata, testdata = data[indices].sample(frac=1), data[~indices].sample(frac=1)


allwordsdict = {}
all_train_utterances = traindata['contents']
all_utterances = list(getalldescriptors(data['contents']))
# print(all_utterances)
le = preprocessing.LabelEncoder() #LabelEncoder for all the possible utterances
transformed_utterances = le.fit_transform(all_utterances)



# # instantiate referent and description encoder
referent_encoder = ReferentEncoder()
description_encoder = DescriptionEncoder(vocab_size=getvocabsize(data['contents']))
choice_ranker = ChoiceRanker("choiceranker", referent_encoder, description_encoder, num_descriptors=len(getalldescriptors(all_train_utterances)), num_referents=len(traindata))
referent_describer = ReferentDescriber(input_dim=len(traindata), num_utterances=len(getalldescriptors(all_train_utterances)))
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


# # Data
NUM_EPOCHS = 10

# def main():
#     # Load in the data

#     # Train the literal models

#     # For each 
#     pass

# def run_pipeline():
#     NUM_LEVEL_RECURSION = 3
#     LEVEL_
#     for i_level in range(NUM_LEVEL_RECURSION):
#         # TODO: define torch optimizer etc. here
#         if i_level == 0:
#             pass
#         else:
#             pass


def train_literal(all_referents, all_descriptors, model):
    # print(all_descriptors)
    """
    model: can be any of the literal listener / literal speaker
    """
    n_train = len(all_referents)
    # turn all_referents and all_descriptors into tensor?
    print(all_referents)
    all_referents = torch.from_numpy(all_referents)
    all_descriptors = torch.from_numpy(all_descriptors)
    # encoded
    encoded_descriptors = [description_encoder.forward(x) for x in all_descriptors]
    encoded_referents = [referent_encoder.forward(x) for x in all_referents]
    # train
    for i_epoch in range(NUM_EPOCHS):
        for i_sample, sample in enumerate(all_referents):
            alternative_ids = np.random.choice(n_train, size=2)
            alternatives = [all_referents[i] for i in alternative_ids]
            batch, batch_ids = alternatives + [sample], alternative_ids + [i_sample]
            encoded_batch = [encoded_referents[i] for i in batch_ids]
            if model.type == "LISTENER":
                # zero the parameters dimension
                optimizer_literal_listener.zero_grad()
                # clarify the dimensions here
                probs = model.forward(encoded_referents=encoded_batch, descriptor_idx=REFERENT_TO_CORRECT_DESCRIPTOR[i_sample], encoded_descriptors=encoded_descriptors)
                # calculate the cross entropy loss
                loss = criterion(probs, labels)
                # update
                loss.backward()
                optimizer_literal_listener.step()
            elif model.type == "SPEAKER":
                # zero the parameters dimension
                optimizer_literal_speaker.zero_grad()
                probs = model.forward(referents=encoded_batch, correct_choice=2, utterances=encoded_descriptors)
                # calculate cross entropy loss
                loss = criterion(probs, labels)
                # update
                loss.backward()
                optimizer_literal_speaker.step()

                

    

if __name__ == "__main__":
    train_literal(traindata['color'], transformed_utterances, l0)

