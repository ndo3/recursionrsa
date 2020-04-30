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
    def __init__(self, name, referentEncoder, stringEncoder, hidden_referent, hidden_descriptor, num_descriptors=261, num_referents=100, hidden_size=100):
        super(ChoiceRanker, self).__init__()
        # metadata
        self.name = name
        self.referentEncoder = referentEncoder
        self.stringEncoder = stringEncoder
        # instantiate the two weight matrices
        self.referentWeights = nn.Linear(hidden_size, hidden_referent) #W4
        self.descriptorWeights = nn.Linear(hidden_size, 1) # description is 1 dimensional
        self.additionalLayer = nn.Linear(hidden_size, num_descriptors)

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

            + () : hidden_size x num_referents 
            | w3 : hidden_size x num_descriptors
            | w3^T : num_descriptors x hidden_size
            w3^T  * () = R: num_descriptors x num_referents
        -=-=-=-=-=-==--=
        referents: num_referents x 3
        descriptors: ,num_descriptors
        labels: ,num_referents
        """
        x = self.referentWeights(referents) + self.descriptorWeights(descriptors)
        # ReLu it
        x = F.relu(x)
        # Then multiply by the additional layer w3
        x = self.additionalLayer(x)
        # Then 2D softmax it
        x_temp = np.zeros((num_descriptors, num_referents))
        for i in range(len(x_temp)):
            x_temp[i] = F.softmax(x[i])
        x = x_temp
        predictions = np.argmax(x, axis=1)
        accuracy = predictions == labels
        return x, accuracy

        
        
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
    def __init__(self, name, referentEncoder, referentDescriber):
        super(LiteralSpeaker, self).__init__()
        self.name = name # adding the name of the model
        self.referentEncoder = referentEncoder
        self.referentDescriber = referentDescriber

    def forward(self, referents, correct_choice):
        return F.softmax(self.referentDescriber(self.referentEncoder(referents)[correct_choice]))  # Outputs a 1d prob dist over utterances

class LiteralListener(nn.Module): #Listener0
    def __init__(self, choice_ranker):
        super(LiteralListener, self).__init__()
        self.choice_ranker = choice_ranker

    def forward(self, all_referents, all_descriptors, labels):
        """
        all_referents: num_referents x 3
        all_descriptions: num_descriptions x 1
        """
        # TODO: Find a way to match up labels with correct referents
        logprobs, accuracy = self.choice_ranker.forward(all_referents, all_descriptors, labels)
        return logprobs, accuracy
        

class ReasoningSpeaker(nn.Module):
    def __init__(self, name, previousListener, literalSpeaker, utterances):
        super(ReasoningSpeaker, self).__init__()
        self.name = name # adding the name of the model to know which level we are at
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
    def __init__(self, name, previousSpeaker):
        super(ReasoningListener, self).__init__()
        self.name = name # adding the name of the model to know which level we are at
        self.previousSpeaker = previousSpeaker

    def forward(self, descriptor_idx, referents):
        # Select the referent that makes the previous speaker most maximize the correct descriptor (descriptor_idx)
        prob_dist = torch.tensor([self.previousSpeaker(referents, i)[descriptor_idx] for i in range(len(referents))])
        prob_dist = F.softmax(prob_dist)

        return prob_dist # Outputs a 1d prob dist over referents