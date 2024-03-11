#This file takes one argument, the path of the file the generated responses need to be written into.


######################################################################
#WARNING: if the file already exists, it will overwrite it.
#####################################################################

import bitsandbytes as bnb
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
BitsAndBytesConfig,
AutoTokenizer,
TrainingArguments,
)
from peft import *
from accelerate import Accelerator
from datasets import load_dataset, list_datasets
from trl import SFTTrainer
import torch
from tqdm import tqdm  # tqdm for displaying progress bars
import pandas as pd
import sys







#Function creating a prompt.
#Takes the data as entry and return a string containing the full prompt
def create_prompt(sample):
  #bos_token = "<s>"
  original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  system_message = "Subjectivity is a characteristic of language: by uttering a statement, the speaker simultaneously expresses his or her position, attitude and feelings towards the statement, thus leaving his or her own mark. If the text is not subjective, it is considered objective. There are a few points to clarify: 1. emotions, considered objective because there is no more objective way of describing them. 2. quotations, considered objctive whatever their content. 3. Intensifiers, considered subjective 4. Speculation, considered subjective. [Answer only OBJ for objective or SUBJ for subjective]."

  #label = sample["label"].replace(original_system_message, "").replace("\n\n### Sentence\n", "").replace("\n### label\n", "").strip()
  input = sample["sentence"]
  #eos_token = "</s>"

  full_prompt = ""
  #full_prompt += bos_token
  full_prompt += "### Instruction:"
  full_prompt += "\n" + system_message
  full_prompt += "\n\n### Input:"
  full_prompt += "\n" + input
  full_prompt += "\n\n### Response:"
  #full_prompt += "\n" + label
  #full_prompt += eos_token

  return full_prompt

#Function generating a response from a model and a prompt
#Takes a model and a prompt as a string as entry
#Return the response as a string
def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.unk_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids)

  return decoded_output[0].replace(prompt, "")


#Function processing a dataset to evaluate a model
#Takes a dataset and a model as entry
#Return a List containing all the responses (as strings)
def process_validation_dataset(dataset, model):
    generated_responses = []

    for sample in tqdm(dataset):
        #print(sample)
        prompt = create_prompt(sample)
        response = generate_response(prompt, model)
        #print(response)
        generated_responses.append(response)

    return generated_responses

if __name__ == '__main__':

    #Data Loading


    data = load_dataset("j03x/CheckThat2023")

    #Data pre-processing
    data['train'] = data['train'].remove_columns(['sentence_id', 'solved_conflict'])
    data['validation'] = data['validation'].remove_columns(['sentence_id', 'solved_conflict'])

    #Unlabeled validation data used for testing Mistral7B without fine-tuning
    data_validation_unlabeled = data['validation'].remove_columns(['label'])


    #Prompt Creation
    p=create_prompt(data_validation_unlabeled[0])


    #Base Model Loading

    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        quantization_config=nf4_config,
        use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id =  tokenizer.unk_token_id
    tokenizer.padding_side = 'left'


    generated_responses = process_validation_dataset(data_validation_unlabeled, model)
    
    
    
    #save into a file
    file_path = sys.argv[0]

    file = open(file_path, 'w')

    for response in generated_responses:
       file.write(response+'\n')
    file.close()




