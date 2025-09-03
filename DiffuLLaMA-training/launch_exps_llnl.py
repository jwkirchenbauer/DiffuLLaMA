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

QOS = "pdebug"
# QOS = "pbatch"

BANK = "guests"
# BANK = "effml"

# TIME_LIMIT = 29
TIME_LIMIT = 59
# TIME_LIMIT = 1440

BASE_OUT_DIR = f"/p/vast1/kirchenb/diffusion-root/DiffuLLaMA/DiffuLLaMA-training/outputs"

BASE_RUN_NAME = f"debug"

WANDB_OFFLINE = False
# WANDB_OFFLINE = True

MODEL_PATH="/p/vast1/pretrain/models/Llama-2-7b-hf"
DATASET_PATH="/p/vast1/pretrain/datasets/diffusion/dolma_v1-6_sample_llama2_pkds"

# MAX_TRAIN_STEPS = 50

# MAX_TRAIN_STEPS = 2067 # 65e9 toks /(16N*4gpn*60mbsz*4accum*2048slen) = 2066.3 steps
# something is fishy here about how App B.2 in https://arxiv.org/pdf/2410.17891 reports this
# versus the args in the repo. Forex, the token bsz is 31M toks per step in this cfg... how/why?
# That said, this would be one way to interpret it I guess, and it could be order 2 days on 16N
MAX_TRAIN_STEPS = 1938 # 65e9 toks /(16N*4gpn*60mbsz*4accum*2048slen) = 1937.1 steps

# static cfgs and then swept params
exp_list = [
    [f"""\
python -u train.py \
--wandb Diffusion \
--seed 2829 \
--max-train-steps {MAX_TRAIN_STEPS}  \
--learning-rate 2e-5  \
--dataset {DATASET_PATH} \
--model {MODEL_PATH} \
--parallel_mode data_parallel \
"""]
]

# GPN = 1
GPN = 4

# nodes
sweep_hparam = [
# 1,
# 2,
# 4,
# 8,
16
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))


# mbsz
sweep_hparam = [
# 1,
# 2,
# 4,
# 8,
16,
# 32,
# 64,
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

# accum
sweep_hparam = [
# 1,
# 2,
# 4,
# 8,
16,
# 64,
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

# seq len
sweep_hparam = [
# 128,
# 1024,
2048,
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))


final_exp_list = exp_list
for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        script_w_args,
        nodes,
        mbsz,
        accum,
        seq_len,
    ) = exp

    # put together the actual "train.py" command
    custom_invocation = f"{script_w_args}"

    # run_name = f"diffusion_llama2-7b_N{nodes}n{nodes*GPN}"
    run_name = f"diffusion_llama2-7b_mb{mbsz}_acc{accum}_sl{seq_len}_N{nodes}n{nodes*GPN}"

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
        --nodes={nodes} \
        --gpus_per_node={GPN} \
        --run_name={run_name} \
        --custom_invocation='{custom_invocation} --batch-size {mbsz} --gradient-accumulate-every {accum} --seq-length {seq_len} --output-dir={BASE_OUT_DIR}/{BASE_RUN_NAME}/{run_name}' \
        --pass_run_name=False \
        {'--dryrun' if WRITE_ONLY else ''}
    """
    total_launches += 1
    if not LIST_CFGS:
        os.system(command)
    else:
        print(run_name)
        # print(command)

print(f"Total launches: {total_launches}")
