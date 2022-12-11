#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ALBERT Model - Final

# 1.Load Dataset

## 1.1 Import torch

# # Install necessary files
# !pip install torch==1.4.0
# !pip install sentencepiece
# !pip install transformers==3.5.1
# !pip install wget

# Instructing PyTorch to use the GPU.
import torch

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('Current GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Releases all unoccupied cached memory 
torch.cuda.empty_cache()

## 1.2 Download Dataset

# The dataset source: https://rajpurkar.github.io/SQuAD-explorer/
import wget
import os

# Setup local directory
print('Downloading dataset...')
local_dir = './squad_dataset/'

# The filenames and URLs for the dataset files.
files = [('train-v1.1.json', 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json'), 
         ('dev-v1.1.json', 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'),
         ('evaluate-v1.1.py', 'https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py')]

# Create directory if needed
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download data-files
for (filename, url) in files:
    file_path = local_dir + filename
    if not os.path.exists(file_path):
        print('  ' + file_path)
        wget.download(url, local_dir + filename)

# Printing file size and location in the drive.
data_dir = './squad_dataset/'
files = list(os.listdir(data_dir))

print('Dataset Location:', data_dir)
for f in files:
    f_size = float(os.stat(data_dir + '/' + f).st_size) / 2**20
    print("     {:25s}    {:>6.2f} MB".format(f, f_size))

## 1.3 Parse Dataset

# The SQuAD dataset is stored in 'json' format. 
# There 87,599 training samples in the dataset.
import json

with open(os.path.join('./squad_dataset/train-v1.1.json'), "r", encoding="utf-8") as reader:
    input_data = json.load(reader)["data"]

# List of dictionary of each row
examples = []

for entry in input_data:
    title = entry["title"] # Extract the title
    # print('  ', title)
    for paragraph in entry["paragraphs"]:
        context_text = paragraph["context"] # Extract the context
        for qa in paragraph["qas"]:
            # Store Question and answer data in dictionary
            ex = {}
            ex['qas_id'] = qa["id"]
            ex['question_text'] = qa["question"]
            answer = qa["answers"][0]
            ex['answer_text'] = answer["text"]
            ex['start_position_character'] = answer["answer_start"]                
            ex['title'] = title
            ex['context_text'] = context_text
            examples.append(ex)

# print('There are {:,} training examples.'.format(len(examples)))

examples[0]

## 1.4 Inspecting Examples:

Each example has a **question**, and a **context**, which is the reference text in which the answer can be found. 


Here are some of the field descriptions from the code:
* **qas_id**: The example's unique identifier
* **title**: Article title
* **question_text**: The question string
* **context_text**: The context string
* **answer_text**: The answer string


import textwrap

wrapper = textwrap.TextWrapper(width=80) 
ex = examples[260]
print('Title:', ex['title'])
print('ID:', ex['qas_id'])

print('\n======== Question =========')
print(ex['question_text'])

print('\n======== Context =========')
print(wrapper.fill(ex['context_text']))

print('\n======== Answer =========')
print(ex['answer_text'])


## 1.5 Helper Functions

import time
import datetime

# Helper function for formatting elapsed times.
# Converts floating point seconds into hh:mm:ss
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Helper function to automatically pick a reasonable interval for printing out a progress update during training.
# For printing updates, this will choose an interval.
def good_update_interval(total_iters, num_desired_updates):
    '''
    Progress update interval based on the magnitude of the total iterations.
    Parameters:
      `total_iters` - The number of iterations in the for-loop.
      `num_desired_updates` - How many times we want to see an update over the 
                              course of the for-loop.
    '''
    exact_interval = total_iters / num_desired_updates
    order_of_mag = len(str(total_iters)) - 1
    round_mag = order_of_mag - 1
    update_interval = int(round(exact_interval, -round_mag))
    if update_interval == 0:
        update_interval = 1
    return update_interval

import pandas as pd
import csv

# Helper function to report current GPU memory usage.
# Reports how much of the GPU's memory we're using.
def check_gpu_mem():
    '''
    Uses Nvidia's SMI tool to check the current GPU memory usage.
    '''
    buf = os.popen('nvidia-smi --query-gpu=memory.total,memory.used --format=csv')
    reader = csv.reader(buf, delimiter=',')
    df = pd.DataFrame(reader)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    return df

# 2.Data Preprocessing




## 2.1 Import Tokenizer

# a row in the dataframe
examples[260]

# Importing the tokenizer
from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# Distributing Sequence Length 
# Choosing max_len
max_len = 384

## 2.2 Tokenizing the training set

import torch

# Time
t0 = time.time()

# Lists
all_input_ids = []
attention_masks = []
segment_ids = [] 
start_positions = []
end_positions = []

num_dropped = 0

# for Update-Interval
update_interval = good_update_interval(total_iters = len(examples), num_desired_updates = 15)

print('Tokenizing {:,} examples...'.format(len(examples)))

for (ex_num, ex) in enumerate(examples):
    # Display update information
    if (ex_num % update_interval) == 0 and not (ex_num == 0):
        elapsed = format_time(time.time() - t0)
        ex_per_sec = (time.time() - t0) / ex_num
        remaining_sec = ex_per_sec * (len(examples) - ex_num)
        remaining = format_time(remaining_sec)
        print('  Example {:>7,}  of  {:>7,}.    Elapsed: {:}. Remaining: {:}'.format(ex_num, len(examples), elapsed, remaining))
    
    answer_tokens = tokenizer.tokenize(ex['answer_text']) # Tokenize the answer
    sentinel_str = ' '.join(['[MASK]']*len(answer_tokens)) # "[MASK] [MASK] [MASK] [MASK] [MASK]"
    start_char_i = ex['start_position_character']
    end_char_i = start_char_i + len(ex['answer_text']) # Compute position of end character
    context_w_sentinel = ex['context_text'][:start_char_i] + sentinel_str + ex['context_text'][end_char_i:] # context-string with sentinel_str in position of answer
    
    # Returns a dictionary containing the encoded sequence or sequence pair and additional information: the mask for sequence classification and the overflowing elements if a max_length is specified.
    encoded_dict = tokenizer.encode_plus(
        ex['question_text'], 
        context_w_sentinel,
        add_special_tokens = True,
        max_length = max_len,
        pad_to_max_length = True,
        truncation = True,
        return_attention_mask = True,
        return_tensors = 'pt')
    
    # They are token indices, numerical representations of tokens building the sequences that will be used as input by the model.
    input_ids = encoded_dict['input_ids']

    # A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT).
    is_mask_token = (input_ids[0] == tokenizer.mask_token_id)
    
    mask_token_indices = is_mask_token.nonzero(as_tuple=False)[:, 0]
    if not len(mask_token_indices) == len(answer_tokens):
        num_dropped += 1
        continue
    
    start_index = mask_token_indices[0]
    end_index = mask_token_indices[-1]
    
    # Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.
    # Returns The tokenized ids of the text.
    answer_token_ids = tokenizer.encode(answer_tokens, 
                                        add_special_tokens=False, 
                                        return_tensors='pt') # Return Pytorch model
    

    input_ids[0, start_index : end_index + 1] = answer_token_ids
    
    all_input_ids.append(input_ids)
    attention_masks.append(encoded_dict['attention_mask'])    
    segment_ids.append(encoded_dict['token_type_ids'])
    start_positions.append(start_index)
    end_positions.append(end_index)

# Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
all_input_ids = torch.cat(all_input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
segment_ids = torch.cat(segment_ids, dim=0)
# Constructs a tensor with no autograd history by copying data
start_positions = torch.tensor(start_positions)
end_positions = torch.tensor(end_positions)

print('DONE.  Tokenization took {:}'.format(format_time(time.time() - t0)))

# 3.Fine-Tuning BERT

## 3.1 Loading Initial Weights

# The AlbertForQuestionAnswering class from the transformers library
from transformers import AlbertForQuestionAnswering
model = AlbertForQuestionAnswering.from_pretrained('albert-base-v2', output_attentions = False, output_hidden_states = False)

desc = model.cuda() # .cuda() Function Can Only Specify GPU.

## 3.2 Sampling and Validation Set


# Represents a Python iterable over a dataset
from torch.utils.data import TensorDataset # Dataset wrapping tensors. Each sample will be retrieved by indexing tensors along the first dimension.
import numpy as np

subsample = True
if subsample:
  # Randomly permute a sequence
    all_indices = np.random.permutation(all_input_ids.shape[0])
    indices = all_indices[0:87000]
    dataset = TensorDataset(all_input_ids[indices, :], 
                            attention_masks[indices, :], 
                            segment_ids[indices, :], 
                            start_positions[indices], 
                            end_positions[indices])
else:
    dataset = TensorDataset(all_input_ids, 
                            attention_masks, 
                            segment_ids, 
                            start_positions, 
                            end_positions)
    
print('Dataset size: {:} samples'.format(len(dataset)))

#This dataset already has a train / test split, but I'm dividing this training set to use 98% for training and 2% for validation

from torch.utils.data import random_split

train_size = int(0.98 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

## 3.3 Batch Size and DataLoaders

from torch.utils.data import DataLoader # Iterable Constructor 
from torch.utils.data import RandomSampler # Samples elements randomly
from torch.utils.data import SubsetRandomSampler # Samples elements randomly from a given list of indices, without replacement
from torch.utils.data import SequentialSampler # Samples elements sequentially, always in the same order

import numpy.random
import numpy as np

batch_size = 12 
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )
validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )
print('{:,} training batches & {:,} validation batches'.format(len(train_dataloader), len(validation_dataloader)))

# Optimizer with fine-tuning recommended
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr = 3e-5, eps = 1e-8)

## 3.4 Epochs and Learning Rate Scheduler

# Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, 
# after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
from transformers import get_linear_schedule_with_warmup

epochs = 2

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

print('Total number of steps: {}'.format(total_steps))

## 3.5 Training Loop

import random
import numpy as np

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []

for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training {:,} batches...'.format(len(train_dataloader)))

    t0 = time.time()
    total_train_loss = 0
    model.train()

    # Setup the update interval
    update_interval = good_update_interval(
                total_iters = len(train_dataloader), 
                num_desired_updates = 15
            )

    num_batches = len(train_dataloader)

    # iterate through each batch
    for step, batch in enumerate(train_dataloader):
        # Display the update interval
        if step % update_interval == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            step_per_sec = (time.time() - t0) / step
            remaining_sec = step_per_sec * (num_batches - step)
            remaining = format_time(remaining_sec)
            print('  Batch {:>7,}  of  {:>7,}.    Elapsed: {:}. Remaining: {:}'.format(step, num_batches, elapsed, remaining))

        # moves the model to the device
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_seg_ids = batch[2].to(device)
        b_start_pos = batch[3].to(device)
        b_end_pos = batch[4].to(device)

        # Sets the gradients of all optimized torch.Tensor s to zero
        model.zero_grad()

        # Ouput
        outputs = model(b_input_ids, 
                        attention_mask=b_input_mask, 
                        token_type_ids = b_seg_ids,
                        start_positions=b_start_pos,
                        end_positions=b_end_pos)
       
        # Output Tuple ( Total span extraction loss is the sum of a Cross-Entropy for the start and end positions, Span-start scores (before SoftMax) , Span-end scores (before SoftMax))
        (loss, start_logits, end_logits) = outputs

        total_train_loss += loss.item() # Returns the value of this tensor as a standard Python number. This only works for tensors with one element. For other cases, see tolist().
        loss.backward() # Computes the gradient of current tensor w.r.t. graph leaves.
        
        # Clips gradient norm of an iterable of parameters.
        # The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        optimizer.step() # method that updates the parameters
        scheduler.step()
    
    # END OF INNER FOR LOOP .........................................................................................


    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    print("")
    print("Running Validation...")

    # In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0

    t0_val = time.time()
    pred_start, pred_end, true_start, true_end = [], [], [], []

    # Compute Validation Metrics
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_seg_ids = batch[2].to(device)
        b_start_pos = batch[3].to(device)
        b_end_pos = batch[4].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, 
                            token_type_ids=b_seg_ids, 
                            attention_mask=b_input_mask,
                            start_positions=b_start_pos,
                            end_positions=b_end_pos)

        (loss, start_logits, end_logits) = outputs        

        total_eval_loss += loss.item()
        start_logits = start_logits.detach().cpu().numpy()
        end_logits = end_logits.detach().cpu().numpy()
      
        b_start_pos = b_start_pos.to('cpu').numpy()
        b_end_pos = b_end_pos.to('cpu').numpy()

        answer_start = np.argmax(start_logits, axis=1)
        answer_end = np.argmax(end_logits, axis=1)

        pred_start.append(answer_start)
        pred_end.append(answer_end)
        true_start.append(b_start_pos)
        true_end.append(b_end_pos)

    pred_start = np.concatenate(pred_start, axis=0)
    pred_end = np.concatenate(pred_end, axis=0)
    true_start = np.concatenate(true_start, axis=0)
    true_end = np.concatenate(true_end, axis=0)

    num_start_correct = np.sum(pred_start == true_start)
    num_end_correct = np.sum(pred_end == true_end)

    total_correct = num_start_correct + num_end_correct
    total_indices = len(true_start) + len(true_end)

    avg_val_accuracy = float(total_correct) / float(total_indices)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0_val)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

## 3.6 Save and Load Model

# from google.colab import drive
# drive.mount('/content/drive')

#import pickle

#pickle.dump(model, open('/content/drive/MyDrive/NLP/albert_model.pkl', 'wb'))

#pickled_model = pickle.load(open('/content/drive/MyDrive/NLP/albert_model.pkl', 'rb'))

## 3.7 Training Results

# Checking for Over-Fitting
import pandas as pd

df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
pd.set_option('precision', 2)
df_stats

# 4.Performance On Test Set

## 4.1 Load Trained & Pre-tuned Model

from google.colab import drive
drive.mount('/content/drive')

import pickle
pickled_model = pickle.load(open('/content/drive/MyDrive/NLP/albert_model.pkl', 'rb'))
model = pickled_model

from transformers import AlbertTokenizer, AlbertForQuestionAnswering

pre_tuned = True

if pre_tuned:
    tokenizer = AlbertTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2")
    model = AlbertForQuestionAnswering.from_pretrained("twmkn9/albert-base-v2-squad2")
    desc = model.cuda()


## 4.2 Parsing Test Set

# highest F1 score that BERT gets among the three is considered
import json

with open(os.path.join('./squad_dataset/dev-v1.1.json'), "r", encoding="utf-8") as reader: input_data = json.load(reader)["data"]

print_count = 0
#print('Unpacking SQuAD Examples...')
#print('Articles:')

examples = []
for entry in input_data:
    title = entry["title"]
    #print('  ', title)
    for paragraph in entry["paragraphs"]:
        context_text = paragraph["context"]
        for qa in paragraph["qas"]:
            ex = {}
            ex['qas_id'] = qa["id"]
            ex['question_text'] = qa["question"]
            ex['answers'] = qa["answers"]
            ex['title'] = title
            ex['context_text'] = context_text
            examples.append(ex)
print('DONE!')

examples[0]

print('There are {:,} test examples.'.format(len(examples)))

## 4.3 Locating Test Answers

# # 2-pass approach
import time
import torch
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

t0 = time.time()

start_positions = []
end_positions = []
num_clipped_answers = 0
num_impossible = 0

# Set the update Interval
update_interval = good_update_interval(
            total_iters = len(examples), 
            num_desired_updates = 15)

print('Processing {:,} examples...'.format(len(examples)))

for (ex_num, ex) in enumerate(examples):

    # Display update-interval information
    if (ex_num % update_interval) == 0 and not (ex_num == 0):
        elapsed = format_time(time.time() - t0)
        ex_per_sec = (time.time() - t0) / ex_num
        remaining_sec = ex_per_sec * (len(examples) - ex_num)
        remaining = format_time(remaining_sec)
        print('  Example {:>7,}  of  {:>7,}.    Elapsed: {:}. Remaining: {:}'.format(ex_num, len(examples), elapsed, remaining))


    start_options = []
    end_options = []

    encoded_stored = False

    for answer in ex['answers']:
        answer_tokens = tokenizer.tokenize(answer['text'])
        sentinel_str = ' '.join(['[MASK]']*len(answer_tokens))
        start_char_i = answer['answer_start']
        end_char_i = start_char_i + len(answer['text'])
        context_w_sentinel = ex['context_text'][:start_char_i] + sentinel_str + ex['context_text'][end_char_i:]
        input_ids = tokenizer.encode(
            ex['question_text'], 
            context_w_sentinel,
            add_special_tokens = True, 
            #max_length = max_len,
            pad_to_max_length = False,
            truncation = False)
        
        mask_token_indices = np.where(np.array(input_ids) == tokenizer.mask_token_id)[0]
        assert(len(mask_token_indices) == len(answer_tokens))           
        start_index = mask_token_indices[0]
        end_index = mask_token_indices[-1]
        start_options.append(start_index)
        end_options.append(end_index)
    
    start_positions.append(start_options)
    end_positions.append(end_options)

print('DONE.  Tokenization took {:}'.format(format_time(time.time() - t0)))

num_impossible = 0
num_clipped = 0

for (start_options, end_options) in zip(start_positions, end_positions):

    is_possible = False
    for i in range(0, len(start_options)):
        if (start_options[i] < max_len) and (end_options[i] < max_len):
            is_possible = True
        if (start_options[i] > max_len) or (end_options[i] > max_len):
            num_clipped += 1
    if not is_possible:
        num_impossible += 1

print('')

print('Samples w/ all answers clipped: {:,} of {:,} ({:.2%})'.format(num_impossible, len(examples), float(num_impossible) / float(len(examples))))

addtl_clipped = num_clipped - (num_impossible * 3)
total_answers = len(examples) * 3
print('\n    Additional clipped answers: {:,} of {:,}'.format(addtl_clipped, total_answers))

## 4.4 Tokenizing and Encoding the  Test Samples

import time
import torch

t0 = time.time()
all_input_ids = []
attention_masks = []
segment_ids = [] 

update_interval = good_update_interval(
            total_iters = len(examples), 
            num_desired_updates = 15
        )

print('Tokenizing {:,} examples...'.format(len(examples)))

for (ex_num, ex) in enumerate(examples):

    if (ex_num % update_interval) == 0 and not (ex_num == 0):
        elapsed = format_time(time.time() - t0)
        ex_per_sec = (time.time() - t0) / ex_num
        remaining_sec = ex_per_sec * (len(examples) - ex_num)
        remaining = format_time(remaining_sec)
        print('  Example {:>7,}  of  {:>7,}.    Elapsed: {:}. Remaining: {:}'.format(ex_num, len(examples), elapsed, remaining))

    encoded_dict = tokenizer.encode_plus(
        ex['question_text'], 
        ex['context_text'],
        add_special_tokens = True,
        max_length = max_len,
        pad_to_max_length = True,
        truncation = True,
        return_attention_mask = True,
        return_tensors = 'pt',
    )
    input_ids = encoded_dict['input_ids']
 
    all_input_ids.append(input_ids)
    attention_masks.append(encoded_dict['attention_mask'])    
    segment_ids.append(encoded_dict['token_type_ids'])
all_input_ids = torch.cat(all_input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
segment_ids = torch.cat(segment_ids, dim=0)

print('DONE.  Tokenization took {:}'.format(format_time(time.time() - t0)))


## 4.5 Evaluate On Test Set

import time
import numpy as np

model.eval()

t0 = time.time()
pred_start = []
pred_end = []
num_test_samples = all_input_ids.shape[0]
batch_size = 16

num_batches = int(np.ceil(num_test_samples / batch_size))

print('Evaluating on {:,} test batches...'.format(num_batches))

batch_num = 0
for start_i in range(0, num_test_samples, batch_size):
    if ((batch_num % 50) == 0) and not (batch_num == 0):
      elapsed = format_time(time.time() - t0)
      batches_per_sec = (time.time() - t0) / batch_num
      remaining_sec = batches_per_sec * (num_batches - batch_num)
      remaining = format_time(remaining_sec)
      print('  Batch {:>7,}  of  {:>7,}.    Elapsed: {:}. Remaining: {:}'.format(batch_num, num_batches, elapsed, remaining))

    end_i = min(start_i + batch_size, num_test_samples)
    b_input_ids = all_input_ids[start_i:end_i, :]
    b_attn_masks = attention_masks[start_i:end_i, :]
    b_seg_ids = segment_ids[start_i:end_i, :]   

    b_input_ids = b_input_ids.to(device)
    b_attn_masks = b_attn_masks.to(device)
    b_seg_ids = b_seg_ids.to(device)

    with torch.no_grad():
        (start_logits, end_logits) = model(b_input_ids, 
                                           attention_mask=b_attn_masks,
                                           token_type_ids=b_seg_ids)
    start_logits = start_logits.detach().cpu().numpy()
    end_logits = end_logits.detach().cpu().numpy()

    answer_start = np.argmax(start_logits, axis=1)
    answer_end = np.argmax(end_logits, axis=1)

    pred_start.append(answer_start)
    pred_end.append(answer_end)

    batch_num += 1

pred_start = np.concatenate(pred_start, axis=0)
pred_end = np.concatenate(pred_end, axis=0)

print('    DONE.')

print('\nEvaluation took {:.0f} seconds.'.format(time.time() - t0))

#5.Results

Exact Match:  Number of  predicted start and end indices that are equal to the correct ones are added up for this metric

total_correct = 0

for i in range(0, len(pred_start)):

    match_options = []
    for j in range (0, len(start_positions[i])):
        matches = 0
        if pred_start[i] == start_positions[i][j]:
            matches += 1
        if pred_end[i] == end_positions[i][j]:
            matches += 1

        match_options.append(matches)

    total_correct += (max(match_options))
total_indices = len(pred_start) + len(pred_end)

print('Correctly predicted indeces: {:,} of {:,} ({:.2%})'.format(
    total_correct,
    total_indices,
    float(total_correct) / float(total_indices)
))

**F1 Score**

precision = 1.0 * num_same / len(pred_toks)

recall = 1.0 * num_same / len(gold_toks)

f1 = (2 * precision * recall) / (precision + recall)

f1s = []
for i in range(0, len(pred_start)):
    pred_span = set(range(pred_start[i], pred_end[i] + 1))
    f1_options = []
    for j in range (0, len(start_positions[i])):
        true_span = set(range(start_positions[i][j], end_positions[i][j] + 1))    
        num_same = len(pred_span.intersection(true_span))
        if num_same == 0:
            f1_options.append(0)
            continue
        precision = float(num_same) / float(len(pred_span))
        recall = float(num_same) / float(len(true_span))
        f1 = (2 * precision * recall) / (precision + recall)
        f1_options.append(f1)
    f1s.append(max(f1_options))

print('Average F1 Score: {:.3f}'.format(np.mean(f1s)))

