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
from accelerate import PartialState 

# TODO: Switch to using dataloader?

class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
            self,
            #model_name: str,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            normalize_embeddings: bool = True,
    ):
        """
        Initializes the embedding function with a specified model from HuggingFace.
        """
        self._tokenizer = tokenizer
        self._model = model
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._normalize_embeddings = normalize_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        return cast(
            Embeddings,
            self._model.encode(
                list(input),
                convert_to_numpy=True,
		normalize_embeddings=self._normalize_embeddings,
                ).tolist()
            )

        self._normalize_embeddings = normalize_embeddings

    def embed_documents(self, documents: List[Documents]) -> List[List[float]]:
        
        # Extract page_content from each Document dictionary
        #print(documents)
        page_contents = [doc for doc in documents]
        #print(page_contents)
        #print(len(page_contents))
        
        # Tokenize the input documents
        encoded_input = self._tokenizer(page_contents, padding=True, truncation=True, return_tensors='pt', max_length=1024)

        # Generate embeddings using the model
        with torch.no_grad():
            model_output = self._model(**encoded_input)

        # Extract embeddings
        hidden_states = model_output.hidden_states
        last_layer_hidden_states = hidden_states[-1]
        embeddings = last_layer_hidden_states.mean(dim=1).tolist()
        #embeddings = model_output.last_hidden_state.mean(dim=1).tolist()
        #print(len(embeddings))
        
        return embeddings

    def embed_query(self, input_data: str) -> List[float]:
        #Converts a list of text documents into embeddings.

        #Parameters:
        #- documents (List[str]): A list of text documents to convert.

        #Returns:
        #- List[List[float]]: A list of embeddings, one per document.

        # Tokenize the input documents. This will turn each document into a format the model can understand.
        encoded_input = self._tokenizer(input_data, padding=True, truncation=True, return_tensors='pt', max_length=1024)

        # Generate embeddings using the model. We'll use the last hidden state for this purpose.
        with torch.no_grad():
            model_output = self._model(**encoded_input)

        # Extract embeddings from the model output. Depending on the model, you might want to adjust this.
        # For many models, taking the mean of the last hidden state across the token dimension is a good starting point.
        #pdb.set_trace()
        #print("model_output:",model_output)
        hidden_states = model_output.hidden_states
        last_layer_hidden_states = hidden_states[-1]
        embeddings = last_layer_hidden_states.mean(dim=1).tolist()
        #embeddings = model_output.last_hidden_state.mean(dim=1).tolist()
        #print(len(embeddings))
        #print(len(embeddings[0]))
        #print(embeddings[0])
        
        return embeddings[0]

def VectorTest(test_filename, retriever, zone, output):
    super_index_list = []
    first_index_list = []
    print("query_keyword\tanswer_key\tsearch_rank")
    with open(test_filename, 'r', encoding='utf-8') as file, open(output,'w') as o:
        for line in file:
            line = line.strip()
            if not line.startswith('#'):
                fields = line.split("\t")
                query = fields[0]
                answer_key = fields[1]
                docs = retriever.get_relevant_documents(query)
                pattern_temp = re.compile(r'(?:^|\W)' + re.escape(str(answer_key)) + r'(?:$|\W)')
                index = 1
                index_list = []
                for doc in docs:
                    #print(re.escape(str(doc.page_content).strip()))
                    if pattern_temp.search(str(doc.page_content).strip()):
                        if len(index_list)==0:
                            first_index_list.append(index)
                        index_list.append(index)
                        super_index_list.append(index)
                        #print("##",doc.page_content)
                    index += 1
                o.write(' '.join([query, answer_key, str(index_list)])+'\n')
                #print("#",docs[0].page_content)
                if len(index_list) == 0:
                    super_index_list.append(zone)
                    first_index_list.append(zone)

    avg_all_rank = sum(super_index_list)/len(super_index_list)
    avg_first_rank = sum(first_index_list)/len(first_index_list)
    with open(output,'a') as o:
        o.write(' '.join(["Average Search Rank:",str(avg_all_rank)])+'\n')
        o.write(' '.join(["Average First Hit Rank:",str(avg_first_rank)])+'\n')
    return (avg_all_rank, avg_first_rank)

# Create the parser
parser = argparse.ArgumentParser(description="--input_file --test_file --model_name")

# Add arguments
parser.add_argument('--input_dir', type=str)
parser.add_argument('--terms_file', type=str)
parser.add_argument('--testing_data', type=str, default="testing_data.txt")
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument('--persist_dir', type=str, default="chroma_db")
parser.add_argument('--test_output', type=str, default="vector_test_output")
parser.add_argument('--adapter_dir', type=str, required=True)

#parser.add_argument('--adapter_dir', type=str, default="TMP_ADAPTER")

args = parser.parse_args()
model_name = args.model_name

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

def load_peft_adapter(repo_id,adapter_filename):
    use_ram_optimized_load=False

    base_model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        # trust_remote_code=True,
        device_map='auto',
    )
    base_model.config.use_cache = False
    base_model.config.output_hidden_states = True

    footprint = base_model.get_memory_footprint()
    print("BASE MEM FOOTPRINT",footprint)

    # Load Lora adapter                     
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_filename,
    )
    peft_model.config.use_cache = False
    peft_model.config.output_hidden_states = True

    return peft_model

base_model_name = model_name
adapter_filename = args.adapter_dir

print("base + adapter model being loaded from")
print("base:   ", base_model_name)
print("adaptor:", adapter_filename)
peft_model = load_peft_adapter(base_model_name,adapter_filename)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

print("base + adapter model loaded", flush=True)
    
#Run VectorDB test: finding relevant documents - on fine-tuned model vectors
#Loading or making Chroma database
embedding_function = LocalEmbeddingFunction(tokenizer,peft_model)
second_chroma_dir = os.path.join(args.input_dir,args.persist_dir)

print(f"Deleting existing directory {args.persist_dir} and creating new ChromaDB.")
if os.path.exists(second_chroma_dir):
    shutil.rmtree(second_chroma_dir)  # Delete the directory if it exists
# Load input files from input_dir
loader = DirectoryLoader(args.input_dir, glob=args.testing_data, loader_cls=TextLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=100)
doc_chunks = text_splitter.split_documents(documents)
print("documents chunked. moving on to vectorization...",flush=True)
vectordb = Chroma.from_documents(documents=doc_chunks, embedding=embedding_function, persist_directory=second_chroma_dir)
print("vectorization complete. moving on to VectorTest...",flush=True)

# Initialize retriever
#zone = 109 #1791
zone = len(vectordb.get()['documents'])
retriever = vectordb.as_retriever(search_kwargs={"k": zone})
#Run VectorDB test: finding relevant documents - on base model vectors
terms_filename = args.terms_file
test_output = args.test_output
test_results = VectorTest(test_filename=terms_filename, retriever=retriever, zone=zone, output=test_output)
#print(test_results)
