#!/bin/bash

#SBATCH --job-name=f_smlg   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=30:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/sm_lg_%j.log  # log file

module load pytorch-gpu/py3/2.1.1
conda activate aa

cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

encoder_dim='976'
attention_type='SummaryMixing'
encoder_module='conformer'
output_folder='results/ft/summary_mix_lg'
hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/summary_mix_lg/1000/save/CKPT+2024-04-19+02-16-03+00
data_folder=/gpfsdswork/dataset/LibriSpeechAsrCorpus
num_encoder_layers=24

python finetune/ft_brq.py finetune/ft_brq.yaml \
    --data_folder $data_folder \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --attention_type $attention_type --encoder_module $encoder_module \
    --num_encoder_layers $num_encoder_layers \
    --output_folder $output_folder


python finetune/ft_brq.py finetune/ft_brq.yaml \
    --data_folder $data_folder \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --attention_type $attention_type --encoder_module $encoder_module \
    --num_encoder_layers $num_encoder_layers \
    --output_folder $output_folder \
    --use_language_modelling true \
    --test_only --kenlm_model_path /gpfswork/rech/nkp/uaj64gk/bestrqexp/4-gram.arpa.gz
