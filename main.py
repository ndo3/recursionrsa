from models import LiteralListener, LiteralSpeaker, ReasoningListener, ReasoningSpeaker
import numpy as np
from scipy.special import logsumexp

# Data
NUM_EPOCHS = 10
REFERENT_ENCODER = ReferentEncoder()
DESCRIPTION_ENCODER = DescriptionEncoder()
REFERENT_TO_CORRECT_DESCRIPTOR = {}



def main():
    # Load in the data

    # Train the literal models

    # For each 
    pass

def run_pipeline():
    NUM_LEVEL_RECURSION = 3
    for i_level in range(NUM_LEVEL_RECURSION):
        pass


def train(all_referents, all_descriptors, model, previous_model=None):
    """
    model: can be any of the literal/pragmatic listener
    """
    n_train = len(all_referents)
    # shuffle
    np.random.shufle(all_referents)
    np.random.shuffle(all_descriptors)
    # encoded
    encoded_descriptors = [DESCRIPTION_ENCODER.forward(x) for x in all_descriptors]
    encoded_referents = [REFERENT_ENCODER.forward(x) for x in all_referents]
    # instantiate accuracy metrics
    acc_count = 0
    # train
    for i_epoch in range(NUM_EPOCHS):
        for i_sample, sample in enumerate(all_referents):
            alternative_ids = np.random.choice(n_train, size=2)
            alternatives = [all_referents[i] for i in alternative_ids]
            batch, batch_ids = alternatives + [sample], alternative_ids + [i_sample]
            encoded_batch = [encoded_referents[i] for i in batch_ids]
            if model.type == "LISTENER":
                # clarify the dimensions here
                probs = model.forward(encoded_referents=encoded_batch, descriptor_idx=REFERENT_TO_CORRECT_DESCRIPTOR[i_sample], encoded_descriptors=encoded_descriptors)
            else if model.type == "SPEAKER":
                probs = model.forward(referents=encoded_batch, correct_choice=2, utterances=encoded_descriptors)
            if not model.reasoning:
                # Take the log probability stuff here - dimensionality should be (num_referents)
                log_sum_exponential = logsumexp(probs)
                deducted = probs - log_sum_exponential
                # Take the argmax here?
                chosen = np.argmax(deducted)
            else:
                chosen = np.argmax(probs)
            if chosen == i_sample: acc_count += 1
        acc = acc_count/n_train
        print("Accuracy: ", acc)
                






    pass
    

if __name__ == "__main__":
    main()