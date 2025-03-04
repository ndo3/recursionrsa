import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from scipy.special import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

DEBUG = False

class ReferentEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100):
        super(ReferentEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x

class DescriptionEncoder(nn.Module):
    def __init__(self, vocab_size=96, embedding_dim=50, hidden_dim=100): # 96 = len(all_descriptors) in main.py
        super(DescriptionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        embeds = self.embeddings(x)
        lstm_out, _ = self.lstm(embeds.view(1, -1, self.embedding_dim))
        lstm_out = lstm_out[-1, :, :] # Only get the last hidden state. We don't care about the others
        x = self.fc1(lstm_out)
        return x


class ChoiceRanker(nn.Module):
    def __init__(self, name, hidden_referent=100, hidden_descriptor=100, num_referents=3, hidden_size=100):
        super(ChoiceRanker, self).__init__()
        # metadata
        self.name = name
        # instantiate the two weight matrices
        self.referentWeights = nn.Linear(hidden_referent, hidden_size) #W4
        self.descriptorWeights = nn.Linear(hidden_descriptor, hidden_size) # W5
        self.additionalLayer = nn.Linear(hidden_size, 1) #W3, this is strange but it's how it's done in the paper

    def forward(self, referents, descriptor):
        """
        Mental map:
            e1: hidden_referent x num_referents
            W4: hidden_size x hidden_referent
            => W4e1: hidden_size x num_referents
            ed: hidden_descriptor x 1
            W5: hidden_size x hidden_descriptor
            => W5ed: hidden_size x 1
            => W4e1 + W5ed :: (hidden_size x (num_referents))

            + () : hidden_size x num_referents
            | w3 : hidden_size x 1
            | w3^T : 1 x hidden_size (1 because we are not doing batch)
            w3^T  * () = R: 1 x num_referents
        -=-=-=-=-=-==--=
        referents: num_referents x 3
        descriptors: ,num_descriptors
        labels: ,num_referents
        """
        x = self.referentWeights(referents) + self.descriptorWeights(descriptor)
        # ReLu it
        x = F.relu(x)
        # Then multiply by the additional layer w3
        x = self.additionalLayer(x)
        x = x.t()
        # Then softmax it
        # Just outputting the end of equation (4) in the paper
        return x[0]



class ReferentDescriber(nn.Module):
    def __init__(self, num_utterances, input_dim=100, hidden_dim=100):
        super(ReferentDescriber, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_utterances)

    def forward(self, referent):
        x = F.relu(self.fc1(referent))
        x = self.fc2(x)

        return x


class LiteralSpeaker(nn.Module):
    def __init__(self, name, referentEncoder, referentDescriber, alpha):
        super(LiteralSpeaker, self).__init__()
        self.name = name # adding the name of the model
        self.referentEncoder = referentEncoder
        self.referentDescriber = referentDescriber
        self.type = "SPEAKER"
        self.reasoning = False
        self.training = True
        self.alpha = alpha
        self.neural = True

    def forward(self, referent, dynamic_dict=None):
        # print("s0 forward")
        encoded_referent = self.referentEncoder(referent)
        out = self.referentDescriber(encoded_referent) * self.alpha  # Outputs a 1d prob dist over utterances
        if not self.training:
            out = F.softmax(out, dim=0)
        return out


class LiteralListener(nn.Module): #Listener0
    def __init__(self, name, choice_ranker, referent_encoder, descriptor_encoder):
        super(LiteralListener, self).__init__()
        self.name = name
        self.choice_ranker = choice_ranker
        self.referent_encoder = referent_encoder
        self.descriptor_encoder = descriptor_encoder
        self.type = "LISTENER"
        self.reasoning = False
        self.training = True

    def forward(self, referents, descriptor, descriptor_idx=None, descriptors=None, dynamic_dict=None):
        # print("l0 forward")
        if self.training:
            encoded_referents = self.referent_encoder(referents)
            encoded_descriptor = self.descriptor_encoder(descriptor)
            x =  self.choice_ranker(encoded_referents, encoded_descriptor) # Outputs a 1d prob dist over referents
        else:
            encoded_referents = self.referent_encoder(referents)
            encoded_descriptor = self.descriptor_encoder(descriptor)
            x =  self.choice_ranker(encoded_referents, encoded_descriptor) # Outputs a 1d prob dist over referents
            x = F.softmax(x, dim=0)
        return x


class ReasoningSpeaker(nn.Module):
    def __init__(self, name, previousListener, literalSpeaker, utterances, alpha):
        super(ReasoningSpeaker, self).__init__()
        self.name = name # adding the name of the model to know which level we are at
        self.previousListener = previousListener
        self.literalSpeaker = literalSpeaker
        self.type = "SPEAKER"
        self.reasoning = True
        self.alpha = alpha

    def get_previous_listener_probs(self, referents, descriptor, descriptor_idx, descriptors, dynamic_dict):
        assert descriptor_idx == descriptor
        if dynamic_dict is None:
            return self.previousListener(referents, descriptor, descriptor_idx, descriptors, dynamic_dict=dynamic_dict)
        if (self.previousListener.name, referents, descriptor_idx) in dynamic_dict:
            # if DEBUG:
            #     print('hit listener')
            return dynamic_dict[(self.previousListener.name, referents, descriptor_idx)]
        else:
            if DEBUG:
                print('missing', self.previousListener.name)
            prob = self.previousListener(referents, descriptor, descriptor_idx, descriptors, dynamic_dict=dynamic_dict)
            dynamic_dict[(self.previousListener.name, referents, descriptor_idx)] = prob
            assert (self.previousListener.name, referents, descriptor_idx) in dynamic_dict
            return prob

    def get_previous_literal_speaker_probs(self, referent, dynamic_dict):
        if dynamic_dict is None:
            return self.literalSpeaker(referent)
        if (self.literalSpeaker.name, tuple(referent.tolist())) in dynamic_dict:
            # if DEBUG:
            #     print('hit literal speaker')
            return dynamic_dict[(self.literalSpeaker.name, tuple(referent.tolist()))]
        else:
            if DEBUG:
                print('missing', self.literalSpeaker.name)
            prob = self.literalSpeaker(referent)
            dynamic_dict[(self.literalSpeaker.name, tuple(referent.tolist()))] = prob
            assert (self.literalSpeaker.name, tuple(referent.tolist())) in dynamic_dict
            return prob

    def forward(self, referents, correct_choice, utterances, dynamic_dict=None):
        referent = referents[correct_choice]
        # Select the utterance that makes the previous listener most maximize the correct referent (correct_choice) regularized by fluency
        # listener_prob_dist = torch.tensor([self.previousListener(referents, descriptor, descriptor_idx=list(utterances).index(descriptor), descriptors=utterances)[correct_choice] for descriptor in utterances], device=device) # Prob(correct choice | descriptor). 1d vector of len num utterances
        listener_prob_dist = torch.tensor([self.get_previous_listener_probs(referents, descriptor, descriptor_idx=list(utterances).index(descriptor), descriptors=utterances, dynamic_dict=dynamic_dict)[correct_choice] for descriptor in utterances], device=device) # Prob(correct choice | descriptor). 1d vector of len num utterances


        # Fluency
        speaker_prob_dist = self.get_previous_literal_speaker_probs(referent, dynamic_dict) # Prob(utterance). 1d vector of len num utterance
        final_scores = listener_prob_dist * speaker_prob_dist * self.alpha# 1d vector of len num utterances.
        final_scores = F.softmax(final_scores, dim=0)
        return final_scores # Outputs a 1d prob dist over utterances


class ReasoningListener(nn.Module):
    def __init__(self, name, previousSpeaker):
        super(ReasoningListener, self).__init__()
        self.name = name # adding the name of the model to know which level we are at
        self.previousSpeaker = previousSpeaker
        self.type = "LISTENER"
        self.reasoning = True

    def get_previous_speaker_probs(self, referents, i, descriptors, dynamic_dict):
        if dynamic_dict is None:
            return self.previousSpeaker(referents, i, descriptors, dynamic_dict=dynamic_dict)
        if (self.previousSpeaker.name, referents, i, descriptors) in dynamic_dict:
            # if DEBUG:
            #     print('hit reasoning speaker')
            return dynamic_dict[(self.previousSpeaker.name, referents, i, descriptors)]
        else:
            if DEBUG:
                print('missing', self.previousSpeaker.name)
            prob = self.previousSpeaker(referents, i, descriptors, dynamic_dict=dynamic_dict)
            dynamic_dict[(self.previousSpeaker.name, referents, i, descriptors)] = prob
            assert (self.previousSpeaker.name, referents, i, descriptors) in dynamic_dict
            return prob

    def get_previous_literal_speaker_probs(self, referent, dynamic_dict):
        if dynamic_dict is None:
            return self.previousSpeaker(referent)
        if (self.previousSpeaker.name, referent) in dynamic_dict:
            # if DEBUG:
            #     print('hit literal speaker')
            return dynamic_dict[(self.previousSpeaker.name, referent)]
        else:
            if DEBUG:
                print('missing', self.previousSpeaker.name)
            prob = self.previousSpeaker(referent)
            dynamic_dict[(self.previousSpeaker.name, referent)] = prob
            assert (self.previousSpeaker.name, referent) in dynamic_dict
            return prob

    def forward(self, referents, descriptor, descriptor_idx=None, descriptors=None, dynamic_dict=None):
        # Select the referent that makes the previous speaker most maximize the correct descriptor (descriptor_idx)
        if self.previousSpeaker.reasoning:
            prob_dist = torch.tensor([self.get_previous_speaker_probs(referents, i, descriptors, dynamic_dict=dynamic_dict)[descriptor_idx] for i in range(len(referents))], device=device)
        else:
            prob_dist = torch.tensor([self.get_previous_literal_speaker_probs(descriptor_idx) for referent in referents], device=device)

        prob_dist = F.softmax(prob_dist, dim=0)

        return prob_dist # Outputs a 1d prob dist over referents


class ClassicLiteralSpeaker:
    def __init__(self, name, alpha):
        self.name = name # adding the name of the model
        self.reasoning = False
        self.type = "SPEAKER"
        self.alpha = alpha
        self.neural = False

    def forward(self, meaning_matrix):
        # first, multiply each row by alpha
        meaning_matrix = np.apply_along_axis(lambda x: x*self.alpha, 1, meaning_matrix)
        # then row wise softmax it (row - colors, for each color softmax all the possible utterances)
        meaning_matrix = np.apply_along_axis(softmax, 1, meaning_matrix)
        # loop through each row and assert that sum has to be 1
        return meaning_matrix


class ClassicLiteralListener:
    def __init__(self, name):
        self.name = name # adding the name of the model
        self.reasoning = False
        self.type = "SPEAKER"
        self.neural = False

    def forward(self, meaning_matrix):
        # first column wise softmax it (column - utterance, softmax all the possible referents)
        meaning_matrix = np.apply_along_axis(softmax, 0, meaning_matrix)
        return meaning_matrix
