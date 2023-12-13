python train_clear.py \
    --model_name_or_path /mnt/hw/nlp/assignment3/model_dl/roberta_base \
    --dataset_name agnews_sup \
    --do_train \
    --do_predict \
    --do_eval \
    --max_seq_length 64 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --output_dir /mnt/hw/nlp/assignment3/outputs/roberta_base/agnews_sup_0/ \
    --evaluation_strategy epoch \
    --save_strategy no \
    --seed 42

python train_clear.py \
    --model_name_or_path /mnt/hw/nlp/assignment3/model_dl/roberta_base \
    --dataset_name agnews_sup \
    --do_train \
    --do_predict \
    --do_eval \
    --max_seq_length 64 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --output_dir /mnt/hw/nlp/assignment3/outputs/roberta_base/agnews_sup_1/ \
    --evaluation_strategy epoch \
    --save_strategy no \
    --seed 3407

python train_clear.py \
    --model_name_or_path /mnt/hw/nlp/assignment3/model_dl/roberta_base \
    --dataset_name agnews_sup \
    --do_train \
    --do_predict \
    --do_eval \
    --max_seq_length 64 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --output_dir /mnt/hw/nlp/assignment3/outputs/roberta_base/agnews_sup_2/ \
    --evaluation_strategy epoch \
    --save_strategy no \
    --seed 60

python train_clear.py \
    --model_name_or_path /mnt/hw/nlp/assignment3/model_dl/roberta_base \
    --dataset_name agnews_sup \
    --do_train \
    --do_predict \
    --do_eval \
    --max_seq_length 64 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --output_dir /mnt/hw/nlp/assignment3/outputs/roberta_base/agnews_sup_3/ \
    --evaluation_strategy epoch \
    --save_strategy no \
    --seed 80

python train_clear.py \
    --model_name_or_path /mnt/hw/nlp/assignment3/model_dl/roberta_base \
    --dataset_name agnews_sup \
    --do_train \
    --do_predict \
    --do_eval \
    --max_seq_length 64 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --output_dir /mnt/hw/nlp/assignment3/outputs/roberta_base/agnews_sup_4/ \
    --evaluation_strategy epoch \
    --save_strategy no \
    --seed 100