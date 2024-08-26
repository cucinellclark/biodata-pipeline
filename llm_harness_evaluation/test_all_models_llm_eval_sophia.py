import subprocess
from subprocess import PIPE
import time,sys,os
import glob

def check_queue(count):
    if count == 5:
        return False
    try:
        cmd = ['qstat','-u','cc8dm']
        status = subprocess.check_output(cmd).decode('utf-8')
        status_parts = status.strip().split('\n')
        # job is already running, dont start one
        # if len(status_parts) > 1:
        if status.count('testmodel') > 2:
            return False 
        return True 
    except Exception as e:
        return check_queue(count+1)
    #p = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    #output, err = p.communicate()
    #print(output.decode('utf-8'))

def write_source_script(model_name, output_file, adapter_dir, task, num_shot, epoch):
    with open('setup_test_model_params_llm_harness_sophia.sh','w') as o:
        o.write(f'MODEL_NAME={model_name}\n')
        o.write(f'OUTPUT_FILE={output_file}\n')
        o.write(f'ADAPTER_DIR={adapter_dir}\n')
        o.write(f'TASK={task}\n')
        o.write(f'NUM_SHOT={num_shot}\n')
        o.write(f'EPOCH={epoch}\n')

model_name = "/lus/eagle/projects/argonne_tpc/chia-llama2/autotrain/Llama-2-7b-chat-hf"
output_prefix = "/lus/eagle/projects/argonne_tpc/chia-clark/BiodataTestingRubric/bvbrc_json_tests/llm_harness/"
epoch_list = list(range(1,5,4))
epoch_list = [str(float(x)) for x in epoch_list]

adapter_type_list = ['bioset_result_1person','bioset_result','genome_amr_1person','genome_amr','genomes_1person','genomes']
test_type = {'winogrande':'5','truthfulqa':'0','arc_challenge':'25','hellaswag':'10','mmlu':'5','gsm8k':'5'}

for adapter_type in adapter_type_list:
    adapter_prefix = f"/lus/eagle/projects/argonne_tpc/cucinell/BiodataTestingRubric/narrative_tests/bvbrc_json_llama3_assignment_test/{adapter_type}_adapters/TMP_RESULTS_Llama-2-7b-chat-hf_"
    for i in range(0,len(epoch_list),8):
        epoch=str(epoch_list[i])    
        # skip epoch if adapter doesnt exist 
        adapter_dir = f"{adapter_prefix}"
        for task in test_type:
            num_shot = test_type[task]
            output_file = f"{output_prefix}{adapter_type}_"
            check_output_file = f"{output_file}{epoch}_{task}_{test_type[task]}.json"
            check_debug_file = f"{output_file}{epoch}_{task}_{test_type[task]}.debug"
            if os.path.exists(check_output_file) or os.path.exists(check_debug_file):
                continue
            while True:
                if check_queue(0):
                    print('setting up environment')
                    write_source_script(model_name, output_file, adapter_dir, task, num_shot, epoch)
                    # submit new job 
                    job_cmd = ['qsub','testmodel_llm_harness_sophia.sh']
                    try:
                        subprocess.check_call(job_cmd)
                        break
                    except Exception as e:
                        print(f'error submitting job: \n{e}\n')
                        sys.exit()
                time.sleep(15)
