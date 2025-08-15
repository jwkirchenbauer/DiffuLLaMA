# fmt: off
import os
from itertools import product, chain

# LIST_CFGS = True
LIST_CFGS = False

# WRITE_ONLY = True
WRITE_ONLY = False

LAUNCHER_FILEPATH = "/p/vast1/$USER/llnl-tools/launch_tuo.py"

RCCL_INSTALL_DIR = (
    "/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib"
)

ROCM_VERSION = "6.3.0"
RCCL_CFG = "rdzv-lbann"

# QOS = "pdebug"
QOS = "pbatch"

# BANK = "guests"
BANK = "effml"

# TIME_LIMIT = 29
TIME_LIMIT = 59
# TIME_LIMIT = 360

BASE_OUT_DIR = f"/p/vast1/kirchenb/diffusion-root/DiffuLLaMA/DiffuLLaMA-training/outputs"

BASE_RUN_NAME = f"debug"

WANDB_OFFLINE = False
# WANDB_OFFLINE = True

# NODES = 1
# GPN = 1
# NODES = 1
# GPN = 4
NODES = 4
GPN = 4

MODEL_PATH="/p/vast1/pretrain/models/Llama-2-7b-hf"
DATASET_PATH="/p/vast1/pretrain/datasets/diffusion/dolma_v1-6_sample_llama2_pkds"

run_name = f"diffusion_llama2-7b_N{NODES}n{NODES*GPN}"

ACCEL_CONFIG="accelerate_configs/multi_node_tuo.yaml"

ACCEL_PREAMBLE=f"accelerate launch \
    --config_file {ACCEL_CONFIG} \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    --num_machines $SLURM_NNODES \
    --num_processes $SLURM_NTASKS \
"""

# Cfgs
exp_list = [
    [f"""\
{ACCEL_PREAMBLE} \
train.py \
--batch-size 60 \
--gradient-accumulate-every 4  \
--seed 2829 \
--wandb Diffusion \
--max-train-steps 20000  \
--learning-rate 1.5e-5  \
--dataset {DATASET_PATH} \
--model {MODEL_PATH}  \
--seq-length 2048 \
--parallel_mode data_parallel \
""", run_name]
]


final_exp_list = exp_list
for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        script,
        run_name,
    ) = exp

    # put together the actual "train.py" command
    custom_invocation = f"{script}"

    # make the complete launcher command
    command = f"""\
    python {LAUNCHER_FILEPATH} \
        --output_dir={BASE_OUT_DIR}/{BASE_RUN_NAME} \
        --wandb_offline={WANDB_OFFLINE} \
        --rocm_version={ROCM_VERSION} \
        --rccl_installdir={RCCL_INSTALL_DIR} \
        --rccl_cfg={RCCL_CFG} \
        --qos={QOS} \
        --bank={BANK} \
        --minutes={TIME_LIMIT} \
        --nodes={NODES} \
        --gpus_per_node={GPN} \
        --run_name={run_name} \
        --custom_invocation='{custom_invocation} --output-dir={BASE_OUT_DIR}/{BASE_RUN_NAME}/{run_name}' \
        --pass_run_name=False \
        {'--dryrun' if WRITE_ONLY else ''}
    """
    total_launches += 1
    if not LIST_CFGS:
        os.system(command)
    else:
        print(run_name)
        print(command)

print(f"Total launches: {total_launches}")
