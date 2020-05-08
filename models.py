import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

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
    def __init__(self, name, referentEncoder, referentDescriber):
        super(LiteralSpeaker, self).__init__()
        self.name = name # adding the name of the model
        self.referentEncoder = referentEncoder
        self.referentDescriber = referentDescriber
        self.type = "SPEAKER"
        self.reasoning = False
        self.training = True

    def forward(self, referent):
        encoded_referent = self.referentEncoder(referent)
        # print("encoded referent size: ", encoded_referent.size())
        # return self.referentDescriber(encoded_referent)  # Outputs a 1d prob dist over utterances
        out = self.referentDescriber(encoded_referent)  # Outputs a 1d prob dist over utterances
        # print("out size: ", out.size())
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

    def forward(self, referents, descriptor, descriptor_idx=None, descriptors=None): # TODO what is this descriptor_idx?
        encoded_referents = self.referent_encoder(referents)
        encoded_descriptor = self.descriptor_encoder(descriptor)
        x =  self.choice_ranker(encoded_referents, encoded_descriptor) # Outputs a 1d prob dist over referents
        if not self.training:
            x = F.softmax(x, dim=0)
        return x
        

class ReasoningSpeaker(nn.Module):
    def __init__(self, name, previousListener, literalSpeaker, utterances):
        super(ReasoningSpeaker, self).__init__()
        self.name = name # adding the name of the model to know which level we are at
        self.previousListener = previousListener
        self.literalSpeaker = literalSpeaker
        self.type = "SPEAKER"
        self.reasoning = True

    def forward(self, referents, correct_choice, utterances):
        # print(self.name + " received forward call")
        referent = referents[correct_choice]
        # Select the utterance that makes the previous listener most maximize the correct referent (correct_choice) regularized by fluency
        listener_prob_dist = torch.tensor([self.previousListener(referents, descriptor, descriptor_idx=list(utterances).index(descriptor), descriptors=utterances)[correct_choice] for descriptor in utterances]) # Prob(correct choice | descriptor). 1d vector of len num utterances

        # Fluency
        speaker_prob_dist = self.literalSpeaker(referent) # Prob(utterance). 1d vector of len num utterance
        final_scores = listener_prob_dist * speaker_prob_dist # 1d vector of len num utterances.
        final_scores = F.softmax(final_scores)
        return final_scores # Outputs a 1d prob dist over utterances


class ReasoningListener(nn.Module):
    def __init__(self, name, previousSpeaker):
        super(ReasoningListener, self).__init__()
        self.name = name # adding the name of the model to know which level we are at
        self.previousSpeaker = previousSpeaker
        self.type = "LISTENER"
        self.reasoning = True

    def forward(self, referents, descriptor, descriptor_idx=None, descriptors=None):
        # Select the referent that makes the previous speaker most maximize the correct descriptor (descriptor_idx)
        print(self.name + " received forward call")
        if self.previousSpeaker.reasoning:
            prob_dist = torch.tensor([self.previousSpeaker(referents, i, torch.tensor(descriptors))[descriptor_idx] for i in range(len(referents))])
        else:
            prob_dist = torch.tensor([self.previousSpeaker(referent)[descriptor_idx] for referent in referents])

        prob_dist = F.softmax(prob_dist)

        return prob_dist # Outputs a 1d prob dist over referents