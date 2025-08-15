# Notes for Tuo

- repo install script should work, just requires base conda like always on Tuo

- they used Tinyllama's data prep...sigh. So, I've copied in our spiritual successors from litgptdev and we can use that to create initial testing data. However near immediate todo will be to integrate the better parquet dataset implementation that we've developed for the retrieval project (based on Jonas' version for recurrent project).


### Data prep 

status: TBD

Needs cpus, just like the installer, so allocate something like

flux alloc -q pbatch --bank=guests --job-name=tokenize -t240 -N1 -n1 -g1 -c96 -ofastload -o mpibind=off --exclusive --unbuffered --label-io

```
export RAW_DATASET_PATH=/p/vast1/pretrain/from_frontier/datasets/fiction/dolma_v1_6-sample/train_shuffled && \
export MODEL_PATH=/p/vast1/pretrain/models/Llama-2-7b-hf && \
export DST_PATH=/p/vast1/pretrain/datasets/diffusion/dolma_v1-6_sample_llama2_pkds && \
python litgptdev_prepare_hf.py \
    --dataset_name_or_path=$RAW_DATASET_PATH \
    --ld_from_disk=True \
    --prefix_type=None \
    --checkpoint_dir=$MODEL_PATH \
    --add_bos=False \
    --add_eos=True \
    --chunk_size=1049088 \
    --skip_remainder=True \
    --destination_path=$DST_PATH \
    --num_proc=32

>
...
INFO:root:Building finished! Took 8.4mins
INFO:root:Dataset written to /p/vast1/pretrain/datasets/diffusion/dolma_v1-6_sample_llama2_pkds
INFO:root:Total chunks/files written across all 70 shards: 8273
INFO:root:Total tokens written across all chunks in all shards: 8.7B
INFO:root:Total trailing separator tokens written across all chunks in all shards: 9
INFO:root:Total skipped tokens across all shards: 40.4M
INFO:root:Packing overhead ratio: 9 / 8.7B = 0.0%
INFO:root:Cleanup cache files returned: 0
```


### Training test 

status: TBD 

automated launch in `launch_exps_llnl.py`

```
1N1n



```


### Orig README below
---

# Overview
Training scripts for training large diffusion language models (e.g., Llama 7B).


## Installation
The code is tested on Python 3.10.12, with several 4xgh200 nodes on a slurm based cluster with The NVIDIA container image for PyTorch, release 24.07, available on [NGC](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-07.html). Since, the container includes several packages not listed in the requirements.txt, we include a pip-freeze of the package list for reference. We only implement the flash-attention version for LLama models. For code without flash-attention, please refer to the Llama factory diffusion adaption code in our repo.

```bash
pip install -r requirements.txt
```

## Data processing

We borrow data processing and dataloaders from TinyLlama. Please preprocess and tokenize the dataset following [them](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md)

## Usage
For multi node runs, we prepare a list of node names in the cluster written in hostnames.txt that the base node can login into via ssh. 
```python
bash multi_node.sh
```
For single node runs, we do not need a list of node names. We directly use the accelerate command in. Note that the command uses the number of nodes to identify the world size, which can be set based on the machine. 
```python
bash run_distributed.sh
```





## Acknowledgements
This work is built on top of the following papers/repositories:
- [Flash-Attention](https://github.com/Dao-AILab/flash-attention)
- [Yunchang](https://github.com/feifeibear/long-context-attention)
- [EasyContext](https://github.com/jzhang38/EasyContext/tree/main)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)


