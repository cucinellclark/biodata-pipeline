import subprocess
from subprocess import PIPE
import time,sys,os
import glob

def check_queue():
    cmd = ['qstat','-u','cc8dm']
    status = subprocess.check_output(cmd).decode('utf-8')
    status_parts = status.strip().split('\n')
    # job is already running, dont start one
    if status.count('debug') > 0:
        return False 
    return True 
    #p = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    #output, err = p.communicate()
    #print(output.decode('utf-8'))

def write_source_script(data_dir, output_dir, model_name, epoch, training_data, prev_adapter, checkpoint):
    with open('setup_model_params.sh','w') as o:
        o.write(f'DATA_DIR={data_dir}\n')
        o.write(f'OUTPUT_DIR={output_dir}\n')
        o.write(f'MODEL_NAME={model_name}\n')
        o.write(f'epoch={epoch}\n')
        o.write(f'training_data={training_data}\n')
        o.write(f'prev_adapter={prev_adapter}\n')
        o.write(f'checkpoint={checkpoint}\n')


adapter_type = ['bioset_result']
for data_type in adapter_type:
    #/lus/eagle/projects/argonne_tpc/chia-clark/DataGeneration/bvbrc_json_narrative_output_llama3/bioset_result
    data_dir = "/lus/eagle/projects/argonne_tpc/chia-clark/DataGeneration/bvbrc_json_narrative_output_llama3"
    output_dir = f"/lus/eagle/projects/argonne_tpc/cucinell/BiodataTestingRubric/narrative_tests/"
    model_name = "/lus/eagle/projects/argonne_tpc/chia-llama2/autotrain/Llama-2-7b-chat-hf"
    #epoch_list = list(range(20,420,20))
    epoch_list = list(range(1,5))
    epoch_list = [str(float(x)) for x in epoch_list]
    #epoch_list = [str(x/100) for x in epoch_list]
    adapter_prefix = f"/lus/eagle/projects/argonne_tpc/cucinell/BiodataTestingRubric/narrative_tests/bvbrc_json_llama3_assignment_test/{data_type}_adapters/TMP_RESULTS_Llama-2-7b-chat-hf_"
    training_data = data_type + '_narrative_llama3.txt'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for idx,epoch in enumerate(epoch_list):
        # skip to next epoch if it exists
        curr_adapter = f"{adapter_prefix}{epoch}" 
        if os.path.exists(curr_adapter):
            continue
        while True:
            if check_queue():
                print('setting up environment')
                # setup source script and pass parameters
                # $1 = data directory, $2 = output directory, $3 = model name, $4 = epoch, $5 = training data 
                # RAG-eval-create_model.py --input_dir $DATA_DIR --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --epochs $epochs --training_data $training_data
                if idx > 0:
                    prev_epoch=epoch_list[idx-1]
                    prev_adapter=f"{adapter_prefix}{prev_epoch}"
                    print(f'prev_adapter = {prev_adapter}')
                    checkpoint_dir = glob.glob(os.path.join(prev_adapter,'checkpoint-*'))[0]
                    if not os.path.exists(checkpoint_dir):
                        print(f'error, checkpoint {checkpoint_dir} doesnt exist: exiting')
                        sys.exit()
                else:
                    prev_adapter='0'
                    checkpoint_dir='0'
                write_source_script(data_dir, output_dir, model_name, epoch, training_data, prev_adapter, checkpoint_dir)
                # submit new job 
                job_cmd = ['qsub','run_epochtest_finetune_modelseed_auto.sh']
                try:
                    subprocess.check_call(job_cmd)
                    break
                except Exception as e:
                    print('error submitting job: \n{e}\n')
                    sys.exit()
            time.sleep(15)
