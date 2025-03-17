import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt                     #for plotting


unk = '<UNK>'     #token for unknown words

def load_test_data(test_data):                          #function to load test data
    with open(test_data, "r", encoding="utf-8") as test_f:
        test = json.load(test_f)
    test_data = []
    for elt in test:
        test_data.append((elt["text"].split(), int(elt["stars"])))  #no subtraction for test data
    return test_data

def write_predictions(model, test_data, word2index, output_file):   #function to write predictions to results/test.out
    model.eval()                                                    #set the model to evaluation mode
    test_data = convert_to_vector_representation(test_data, word2index)  #vectorize test data
    predictions = []

    with torch.no_grad():                           #disable gradient computation for evaluation
        for input_vector, _ in test_data:
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector).item() #get the predicted label
            predictions.append(predicted_label + 1)               #convert back to 1-5 star rating

    with open(output_file, "w") as f:                             #write predictions to the output file
        for prediction in predictions:
            f.write(f"{prediction}\n") 

# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):                            
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)   # W1: 1st linear layer mapping input to hidden dimension
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim) # W2: 2nd linear layer mapping hidden dimension to output dimension

        self.softmax = nn.LogSoftmax(dim=0) # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):   # Computes loss between predicted and true labels
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):  #defines forward pass of the network
        # [to fill] obtain first hidden layer representation
        hidden_layer = self.activation(self.W1(input_vector))
        # [to fill] obtain output layer representation
        output_layer = self.W2(hidden_layer)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output_layer)
        return predicted_vector

# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):           #creates vocabulary from the training data
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 

# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):        # Maps words to indices and vice versa
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 

# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):     #converts text data into bag-of-words vectors
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

def load_data(train_data, val_data):                #loads and preprocesses training and validation data
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":                          # Parses command-line arguments for hyperparameters and file paths
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", required = True, help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    #fix random seeds (for reproducibility)
    random.seed(42)
    torch.manual_seed(42)

    #load data and create vocabulary
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    test_data = load_test_data(args.test_data)  # Load test data
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    #convert text data into numerical vectors
    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)           # Creates the FFNN model
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)     # Creates the SGD optimizer  #default lr = 0.01

    train_losses = []           #save training loss (bonus)
    valid_accuracies = []       #save validation accuracy (bonus)

    print("========== Training for {} epochs ==========".format(args.epochs))   # Trains the model using mini-batches
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        #evaluates the model on the validation set
        misclassified = []      #create an array for misclassified examples (bonus)
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16 
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
 
        #Bonus: get misclassified examples
        for minibatch_index in tqdm(range(len(valid_data) // minibatch_size)): #traverse minibatches
            for example_index in range(minibatch_size):
                idx = minibatch_index * minibatch_size + example_index  #ensures batch-based indexing

                if idx >= len(valid_data):
                    print(f"Index {idx} out of bounds!")  #debugging statement
                    break  

                input_vector, gold_label = valid_data[idx]  #get input features and correct label for current example

                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)

                if predicted_label != gold_label:                       #get only misclassified examples
                    words = input_vector.nonzero().squeeze().tolist()   #make list from the input vector
                    if not isinstance(words, list):                     #ensure words is always a list 
                        words = [words]                                 

                    words_str = [index2word.get(w, "UNK") for w in words] #obtain words from indices
                    print(f"Misclassified Example {idx}:")      
                    print(f"   Words: {words_str}")
                    print(f"   Predicted Label: {predicted_label.item()}")
                    print(f"   Actual Label: {gold_label}\n")

                    misclassified.append((tuple(words_str), predicted_label.item() + 1, gold_label + 1)) #append current example to misclassified array

        train_losses.append(loss.item())  #store final training loss for the epoch (bonus)
        valid_accuracies.append(correct / total)  #store final validation accuracy for the epoch (bonus)

    epochs = range(1, len(train_losses) + 1)    #epochs based on training loss values (bonus)

    plt.figure(figsize=(8, 5))                  #Bonus: plot figure definition 
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')   #training loss plot
    plt.plot(epochs, valid_accuracies, label='Validation Accuracy', marker='s') #validation accuracy plot
    plt.xlabel('Epoch')                                                             
    plt.ylabel('Loss / Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    plt.show()

    #write out to results/test.out
    os.makedirs("results", exist_ok=True)
    write_predictions(model, test_data, word2index, "results/test.out")
    print("Predictions written to results/test.out")
    