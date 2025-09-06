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

# INDUCTOR_CACHE=None
INDUCTOR_CACHE="/l/ssd/$USER"

# EXTRA_COMPILE_FLAGS = False
EXTRA_COMPILE_FLAGS = True

# LOG_RECOMPILES=False
LOG_RECOMPILES = True

ROCM_VERSION = "6.3.0"
RCCL_CFG = "rdzv-lbann"

# QOS = "pdebug"
QOS = "pbatch"

# BANK = "guests"
BANK = "effml"

# TIME_LIMIT = 29
# TIME_LIMIT = 59
TIME_LIMIT = 1440

# REPETITIONS = 1
# DEPENDENCY = None
REPETITIONS = 3
DEPENDENCY = "afterany"

BASE_OUT_DIR = f"/p/vast1/kirchenb/diffusion-root/DiffuLLaMA/DiffuLLaMA-training/outputs"

BASE_RUN_NAME = f"debug"
# BASE_RUN_NAME = f"compile_test"

WANDB_OFFLINE = False
# WANDB_OFFLINE = True

MODEL_PATH="/p/vast1/pretrain/models/Llama-2-7b-hf"
DATASET_PATH="/p/vast1/pretrain/datasets/diffusion/dolma_v1-6_sample_llama2_pkds"

# MAX_TRAIN_STEPS = 2067 # 65e9 toks /(16N*4gpn*60mbsz*4accum*2048slen) = 2066.3 steps
# something is fishy here about how App B.2 in https://arxiv.org/pdf/2410.17891 reports this
# versus the args in the repo. Forex, the token bsz is 31M toks per step in this cfg... how/why?
# That said, this would be one way to interpret it I guess, and it could be order 2 days on 16N
# MAX_TRAIN_STEPS = 1938 # 65e9 toks /(16N*4gpn*60mbsz*4accum*2048slen) = 1937.1 steps

# Okay! DiffuCoder specs make much more sense:
# "The training was conducted on 10 nodes of 8 H100 GPUs each, using BF16 and full-shard
# FSDP (Zhao et al., 2023). The total wall-clock time for training on 65B tokens (100,000 global steps)
# was approximately 40 hours. The single-GPU batch size was 2, and the context window was 4096.
# Following LLaDA (Nie et al., 2024), we truncated 1% of the data to a random length to improve
# handling of variable-length inputs. Additionally, for another 1% of the data, we used a random-length
# prefix as the condition during the diffusion process, which was kept unnoised. We used the Adam
# optimizer with a maximum learning rate of 2e-5, with a linear warmup of 20,000 steps followed by
# a cosine decay schedule to 10% of its peak value at the end of training. The attention mask annealing
# was performed over 10,000 steps, following DiffuLLaMA"

# Their reported world batch size is 10N*8gpn*2*4096 = 655,360 toks , more reasonable than 31M
# a reasonable solve for our devices is 655,360 / 4096 / 10mbsz = 16 devices, so 4 nodes here
# Just to check, 65e9 / 655,360 = 99,182.13 -> "100k global steps" yay

# Now some painful testing revealed that actually, the small mbsz of 2 is required to make the speeds
# work out, along with gradient checkpointing (torch compile didnt work but that would have helped too).
# So the 80 device setup can be achieved with 20N * 4pgn * 2 mbsz * 4096 slen = 655,360 toks
# and while the 40hrs / 100k steps implies they got 40*60*60 / 100000 = 1.44 sec / step
# best I can see is 2.2 - 2.5 sec/step so that is 60-70 hrs. But that is in the realm of feasibility.

MAX_TRAIN_STEPS = 100_000
WARMUP_STEPS = 20_000
PEAK_LR = 2e-5
MIN_LR = 2e-6

# # debug
# MAX_TRAIN_STEPS = 20
# WARMUP_STEPS = 5
# PEAK_LR = 2e-5
# MIN_LR = 2e-9

# static cfgs and then swept params
exp_list = [
    [f"""\
python -u train.py \
--wandb Diffusion \
--seed 2829 \
--max-train-steps {MAX_TRAIN_STEPS}  \
--warmup-steps {WARMUP_STEPS}  \
--learning-rate {PEAK_LR}  \
--min-learning-rate {MIN_LR}  \
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
# 16
# 10
20
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))


# mbsz
sweep_hparam = [
# 1,
2,
# 4,
# 8,
# 10,
# 16,
# 32,
# 64,
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

# accum
sweep_hparam = [
1,
# 2,
# 4,
# 8,
# 16,
# 64,
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

# seq len
sweep_hparam = [
# 128,
# 1024,
# 2048,
4096,
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
        --cache_dir={INDUCTOR_CACHE} \
        --add_compile_flags={EXTRA_COMPILE_FLAGS} \
        --log_recompiles={LOG_RECOMPILES} \
        --qos={QOS} \
        --bank={BANK} \
        --repetitions={REPETITIONS}{f' --dependency={DEPENDENCY}' if DEPENDENCY is not None else ''} \
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
