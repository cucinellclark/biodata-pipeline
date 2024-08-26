import subprocess
from subprocess import PIPE
import time,sys,os
import glob

user = 'cc8dm'

def check_queue():
    cmd = ['qstat','-u',user]
    status = subprocess.check_output(cmd).decode('utf-8')
    status_parts = status.strip().split('\n')
    # job is already running, dont start one
    if len(status_parts) > 1:
        return False 
    return True 

def write_source_script(input_dir, terms_file, testing_data, model_name, chroma_dir, output_file, adapter_dir):
    with open('setup_test_model_params.sh','w') as o:
        o.write(f'INPUT_DIR={input_dir}\n')
        o.write(f'TERMS_FILE={terms_file}\n')
        o.write(f'TESTING_DATA={testing_data}\n')
        o.write(f'MODEL_NAME={model_name}\n')
        o.write(f'CHROMA_DIR={chroma_dir}\n')
        o.write(f'OUTPUT_FILE={output_file}\n')
        o.write(f'ADAPTER_DIR={adapter_dir}\n')

input_dir = "/lus/eagle/projects/argonne_tpc/chia-llama2/RAG-evaluation/data_modelseed"
terms_file = "/home/cc8dm/RAG-eval/tests_small/name2name.txt"
testing_data = "testing_data.cpds " # joined with input_dir
model_name = "/lus/eagle/projects/argonne_tpc/chia-llama2/autotrain/Llama-2-7b-chat-hf"
chroma_prefix = "/home/cc8dm/RAG-eval/GeneratedData/mixtral_gen1_output/chroma_persist_cpds"
output_prefix = "/home/cc8dm/RAG-eval/GeneratedData/mixtral_gen1_output_mistral/cpds_test_output/finetune_llama2_"
adapter_prefix = "/lus/eagle/projects/argonne_tpc/cucinell/GeneratedData/MixtralGen1/TMP_RESULTS_Llama-2-7b-chat-hf_"
epoch_list = list(range(50,1050,50))
epoch_list = [str(float(x/100)) for x in epoch_list]

for idx,epoch in enumerate(epoch_list):
    # skip epoch if adapter doesnt exist 
    adapter_dir = f"{adapter_prefix}{epoch}" 
    chroma_dir = f"{chroma_prefix}"
    output_file = f"{output_prefix}{epoch}"
    if not os.path.exists(adapter_dir):
        print(f'Adapter directory {adapter_dir} does not exist, skipping')
        continue
    while True:
        if check_queue():
            print('setting up environment')
            write_source_script(input_dir, terms_file, testing_data, model_name, chroma_dir, output_file, adapter_dir)
            # submit new job 
            job_cmd = ['qsub','run_epochtest_testmodel_modelseed_auto.sh']
            try:
                subprocess.check_call(job_cmd)
                break
            except Exception as e:
                print('error submitting job: \n{e}\n')
                sys.exit()
        time.sleep(15)
