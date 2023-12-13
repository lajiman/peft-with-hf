# python train_clear.py \
#     --model_name_or_path /mnt/hw/nlp/assignment3/model_dl/roberta_large \
#     --peft_name adapter \
#     --dataset_name restaurant_sup \
#     --do_train \
#     --do_predict \
#     --do_eval \
#     --max_seq_length 64 \
#     --per_device_train_batch_size 128 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 50 \
#     --output_dir /mnt/hw/nlp/assignment3/outputs/adapter_roberta_large/restaurant_sup_0/ \
#     --overwrite_output \
#     --evaluation_strategy epoch \
#     --save_strategy no \

python train_clear.py \
    --model_name_or_path /mnt/hw/nlp/assignment3/model_dl/roberta_large \
    --peft_name adapter \
    --dataset_name acl_sup \
    --do_train \
    --do_predict \
    --do_eval \
    --max_seq_length 64 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 50 \
    --output_dir /mnt/hw/nlp/assignment3/outputs/adapter_roberta_large/acl_sup_0/ \
    --overwrite_output \
    --evaluation_strategy epoch \
    --save_strategy no 


python train_clear.py \
    --model_name_or_path /mnt/hw/nlp/assignment3/model_dl/roberta_large \
    --peft_name adapter \
    --dataset_name agnews_sup \
    --do_train \
    --do_predict \
    --do_eval \
    --max_seq_length 64 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 50 \
    --output_dir /mnt/hw/nlp/assignment3/outputs/adapter_roberta_large/agnews_sup_0/ \
    --overwrite_output \
    --evaluation_strategy epoch \
    --save_strategy no 
