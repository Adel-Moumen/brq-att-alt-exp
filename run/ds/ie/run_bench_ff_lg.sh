#!/bin/bash
#SBATCH --job-name=er_fflg   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=100:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/ff_lg_er_%j.log  # log file


module load pytorch-gpu/py3/2.1.1
conda activate aa
cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/old/ff_lg/1000/save/CKPT+2024-02-07+11-51-51+00
num_layers='25'
num_encoder_layers='24'
encoder_dim='768' # change to ???
attention_type='fastattention'
encoder_module='conformer'
output_folder='results/MP3/ff_lg'
benchmark_location=/gpfswork/rech/nkp/uaj64gk/attention_alt/benchmarks

task='IEMOCAP'
downstream='ecapa_tdnn'

for i in {1..10}
do
echo "on speaker $i"
python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
    --data_folder /users/rwhetten/IEMOCAP/IEMOCAP_full_release \
	--num_layers_ssl "$num_layers" --num_encoder_layers "$num_encoder_layers" --ssl_hub "$hub" --encoder_dim "$encoder_dim" --output_folder "$output_folder"/"$task"/"$downstream/$i" \
	--attention_type "$attention_type" --encoder_module "$encoder_module" --test_spk_id=$i
done

downstream='linear'

for i in {1..10}
do
echo "on speaker $i"
python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
    --data_folder /users/rwhetten/IEMOCAP/IEMOCAP_full_release \
	--num_layers_ssl "$num_layers" --num_encoder_layers "$num_encoder_layers" --ssl_hub "$hub" --encoder_dim "$encoder_dim" --output_folder "$output_folder"/"$task"/"$downstream/$i" \
	--attention_type "$attention_type" --encoder_module "$encoder_module" --test_spk_id=$i
done