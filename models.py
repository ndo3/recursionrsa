import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class ReferentEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100):
        super(ReferentEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x

class DescriptionEncoder(nn.Module):
    def __init__(self, vocab_size=100, embedding_dim=50, hidden_dim=100):
        super(DescriptionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        embeds = self.embeddings(x)
        lstm_out, _ = self.lstm(embeds.view(len(x), 1, -1))
        x = self.fc1(lstm_out)
        return x


class ChoiceRanker(nn.Module):
    def __init__(self, model_name, referentEncoder, stringEncoder, hidden_referent, hidden_descriptor, num_descriptor=261, num_referents=100, hidden_size=100):
        super(ChoiceRanker, self).__init__()
        # metadata
        self.modelName = modelName
        self.referentEncoder = referentEncoder
        self.stringEncoder = stringEncoder
        # instantiate the two weight matrices
        self.referentWeights = nn.Linear(hidden_size, hidden_referent) #W4
        self.descriptorWeights = nn.Linear(hidden_size, 1) # description is 1 dimensional
        self.additionalLayer = nn.Linear(hidden_size)

        

    def forward(self, referents, descriptors, labels, prefix=""):
        """
        Mental map:
            e1: hidden_referent x num_referents
            W4: hidden_size x hidden_referent 
            => W4e1: hidden_size x num_referents
            ed: hidden_descriptor x 1
            W5: hidden_size x hidden_descriptor
            => W5e1: hidden_size x 1
            => W4e1 + W5ed :: (hidden_size x (num_referents))
            w3 : hidden_size x 1
            R: 1 x num_referents
        -=-=-=-=-=-==--=
        referents: 
        """
        x = self.referentWeights(referents) + self.descriptorWeights(descriptors)
        # ReLu it
        x = F.relu(x)
        # Then multiply by 
        x = self.additionalLayer(x)

        
        
        
        

class ReferentDescriber(nn.Module):
    def __init__(self, input_dim=100, num_utterances=261, hidden_dim=100):
        super(ReferentDescriber, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_utterances)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = self.f2(x)
        
        return x

class LiteralSpeaker(nn.Module):
    def __init__(self, referentEncoder, referentDescriber):
        super(LiteralSpeaker, self).__init__()
        self.referentEncoder = referentEncoder
        self.referentDescriber = referentDescriber

    def forward(self, referents, correct_choice):
        return F.softmax(self.referentDescriber(self.referentEncoder(referents)[correct_choice]))  # Outputs a 1d prob dist over utterances

class LiteralListener(nn.Module):
    def __init__(self):
        super(LiteralListener, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReasoningSpeaker(nn.Module):
    def __init__(self, previousListener, literalSpeaker, utterances):
        super(ReasoningSpeaker, self).__init__()
        self.previousListener = previousListener
        self.literalSpeaker = literalSpeaker
        self.utterances = utterances

    def forward(self, referents, correct_choice):
        referent = referents[correct_choice]
        # Select the utterance that makes the previous listener most maximize the correct referent (correct_choice) regularized by fluency
        listener_prob_dist = torch.tensor([self.previousListener(descriptor, referents)[correct_choice] for descriptor in self.utterances]) # Prob(correct choice | descriptor). 1d vector of len num utterances

        # Fluency
        speaker_prob_dist = self.literalSpeaker(referent) # Prob(utterance). 1d vector of len num utterances

        final_scores = listener_prob_dist * speaker_prob_dist # 1d vector of len num utterances.
        final_scores = F.softmax(final_scores)
        return final_scores # Outputs a 1d prob dist over utterances

class ReasoningListener(nn.Module):
    def __init__(self, previousSpeaker):
        super(ReasoningListener, self).__init__()
        self.previousSpeaker = previousSpeaker

    def forward(self, descriptor_idx, referents):
        # Select the referent that makes the previous speaker most maximize the correct descriptor (descriptor_idx)
        prob_dist = torch.tensor([self.previousSpeaker(referents, i)[descriptor_idx] for i in range(len(referents))])
        prob_dist = F.softmax(prob_dist)

        return prob_dist # Outputs a 1d prob dist over referents