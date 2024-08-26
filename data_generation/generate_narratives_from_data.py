import csv
import sys
import glob
import os
import requests
import json
import textwrap
import random
import time
import argparse
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# https://github.com/argonne-lcf/inference-endpoints/tree/main
# https://github.com/argonne-lcf/inference-endpoints/blob/main/remote_inference_gateway.ipynb
with open('access_token.txt','r') as i:
    access_token = i.read()
access_token = access_token.strip()
openai_api_key = access_token
openai_api_base = "https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1"
model = "meta-llama/Meta-Llama-3-70B-Instruct"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def generate_prompt(genome_data):
    prompt = """
        For the json record provided below, write a textual description of the record as a short paragraph that can be included in a textbook or a scientific report.
        Make the description factual, direct, informative, and comprehensive using maximum information provided in the record.
        Do not calculate values that do not exist explicitly in the record.
        Always end the response with ### followed by a new line.
    """
    prompt += json.dumps(genome_data)
    return prompt

def get_random_subset(size, input_list):
    return random.sample(input_list, size)

def run_query(prompt):
    message = {
        "role": "user",
        "content": prompt
    }
    response = client.chat.completions.create(
        model=model,
        messages=[message],
        temperature=1.0
    )
    res_json = json.loads(response.server_response)
    return res_json['choices'][0]['message']['content']

def extract_text_without_pattern(text, pattern):
    pattern_re = re.compile(rf"{pattern}\s*$")
    match = pattern_re.search(text)
    if match:
        return text[:match.start()]
    else:
        return text

def send_requests_and_save_responses(entry,outfile):
    if os.path.exists(outfile):
        return None
    try:
        prompt = generate_prompt(entry)
        content = run_query(prompt)
        if '###' in content[-10:]:
            content = extract_text_without_pattern(content, '###')
            with open(outfile,'w') as o:
                o.write(content)
        else:
            return None
    except (requests.exceptions.RequestException, Exception) as e:
        print(f"Error occurred while processing entry: {e}")
        return None

def get_file_entries(input_file):
    data = []
    with open(input_file, 'r') as i:
        for line in i:
            if len(line.strip()) > 2:
                data.append(line)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='a single data file')
    parser.add_argument('-o', '--output_prefix', help='output file', default='output')

    args = parser.parse_args()

    file_data = get_file_entries(args.file)
    suffix_list = list(range(0,len(file_data)))
    output_list = [args.output_prefix + '_' + str(x) + '.txt' for x in suffix_list]
    response_list = []

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(send_requests_and_save_responses, entry, outfile) for entry,outfile in zip(file_data,output_list)]
        for future in futures:
            output = future.result()
            if output:
                response_list.append(output)

