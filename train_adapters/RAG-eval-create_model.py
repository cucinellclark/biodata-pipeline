from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
import torch
import torch.nn.functional as F
import argparse
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
import json
from typing import List,Union
from chromadb.api.types import (
    Document,
    Documents,
    Embedding,
    Image,
    Images,
    EmbeddingFunction,
    Embeddings,
    is_image,
    is_document,
)
import re,shutil
#import pdb
#Fine-tune model
from datasets import Dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from datasets import load_dataset
from transformers import TrainingArguments, pipeline
from trl import SFTTrainer
from accelerate import PartialState, Accelerator 
import sys

#Create the parser
parser = argparse.ArgumentParser(description="--input_file --test_file --model_name")

# Add arguments
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument('--epochs', type=float, default=1.0)
parser.add_argument('--training_data', type=str,required=True)
parser.add_argument('--model_checkpoint',type=str)
parser.add_argument('--trainer_checkpoint',type=str)

#parser.add_argument('--adapter_dir', type=str, default="TMP_ADAPTER")

args = parser.parse_args()
model_name = args.model_name

if not args.output_dir:
    args.output_dir = args.input_dir

def get_text_data(filename):
    instruction = "Learn this biology information. "   # same for every line here                                                                               
    list_of_text_dicts = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            text = ("### Instruction: \n" + instruction + "\n" +
                    "### Input: \n" + line + "\n" +
                    "### Response :\n" + line)
            list_of_text_dicts.append( { "text": text } )
    return list_of_text_dicts

def user_prompt(human_prompt):
    # must chg if dataset isn't formatted as Alpaca
    prompt_template=f"### HUMAN:\n{human_prompt}\n\n### RESPONSE:\n"
    return prompt_template

base_model_name = model_name
#adapter_filename = os.path.join(args.output_dir,f"TMP_adapter_{os.path.basename(base_model_name)}_{args.epochs}")
accelerator = Accelerator()
device_string = PartialState().process_index # accelerate
if args.model_checkpoint:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        #load_in_8bit=True,
        #load_in_4bit=True,
        #quantization_config=bnb_config,
        device_map={'':device_string}, # accelerate
        #device_map='auto',
    )
    base_model.config.use_cache = False
    #print(f'loading peft model: \n{base_model}\n{args.model_checkpoint}\n')
    model = PeftModel.from_pretrained(base_model,args.model_checkpoint,is_trainable=True)
    footprint = model.get_memory_footprint()
    print("BASE MEM FOOTPRINT",footprint)
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        #load_in_8bit=True,
        #load_in_4bit=True,
        #quantization_config=bnb_config,
        device_map={'':device_string}, # accelerate
        #device_map='auto',
    )
    base_model.config.use_cache = False
    print(base_model)

    footprint = base_model.get_memory_footprint()
    print("BASE MEM FOOTPRINT",footprint)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        # target modules varies from model to model
        #target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # llama2-7b
        #target_modules=["c_attn","c_proj"], # gpt2
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, lora_config)

filename = os.path.join(args.input_dir, args.training_data)
print(filename)
data = get_text_data(filename)

train_dataset = Dataset.from_dict({key: [dic[key] for dic in data] for key in data[0]})
print(train_dataset)
print(train_dataset[0])
tmp_results_output_dir = os.path.join(args.output_dir,f"TMP_RESULTS_{os.path.basename(base_model_name)}_{args.epochs}")

training_arguments = TrainingArguments(
    output_dir = tmp_results_output_dir,
    #per_device_train_batch_size = 16,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=args.epochs,
    #max_steps=8000,
    learning_rate=2e-4,
    fp16=True, #False on a mac
    warmup_ratio = 0.03,
    group_by_length=True,
    lr_scheduler_type = "linear",  # vs constant
    report_to = "none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    args=training_arguments,
    max_seq_length=1024
)

trainer.accelerator.print(f"{trainer.model}")
trainer.model.print_trainable_parameters()
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy
    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

if args.trainer_checkpoint:
    trainer.train(args.trainer_checkpoint)
else:
    trainer.train()

#trainer.save_model(adapter_filename)
trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()
'''
model.save_pretrained(
        adapter_filename,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model)
)
'''
