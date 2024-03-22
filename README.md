# PEE-KD

Code and data for Harnessing the Power of Prompt Experts: Efficient Knowledge Distillation for Enhanced Language Understanding （ECML-PKDD 24）

The key idea is to use different task-specific trainable prompts to extract different views of samples for supervising student models.
We further employ an uncertainty-based mechanism and a selector module to increase robustness and correctness. 

### Setup
We recommand to setup the running enviroment via conda:

```shell
conda create -n pee python=3.8.5
conda activate pee
conda install -n pt2 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

### Data
For SuperGLUE datasets, we download them from the Huggingface Datasets APIs (embedded in our codes).

### Training
Run training scripts in [run_script](run_script) (e.g., RoBERTa for RTE):

```shell
bash run_script/run_rte_roberta.sh
```

### Distillation
Specify the teacher path in the run_kd.sh and execute the script for training the student model:

```shell
bash run_script/run_rte_roberta.sh
```


