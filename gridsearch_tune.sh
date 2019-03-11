#!/bin/bash

# python run_commonsense_qa_debug.py --task_name commonsenseqa --do_train --do_eval --data_dir
# /data/m1/huangl7/CommonsenseQaPlus/baselines/data/ --bert_model bert-base-uncased --max_seq_length 128
# --train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir tmp5/

task="commonsenseqa"

batchsizes=( 2 4 8 16 32 )

for s in "${batchsizes[@]}"
do
    learningrates=( 2e-5 3e-5 5e-5 )

    for l in "${learningrates[@]}"
    do
        epochs=( 3 4 5 )

        for e in "${epochs[@]}"
        do
            python run_commonsense_qa_debug.py --task_name "${task}" --do_eval --do_train --bert_model bert-large-uncased --data_dir /data/m1/huangl7/CommonsenseQaPlus/baselines/data/ --max_seq_length 128 --train_batch_size ${s} --learning_rate ${l} --num_train_epochs ${e} --output_dir /data/m1/huangl7/CommonsenseQaPlus/baselines/pytorch_pretrained_bert/output/batch_${s}_lr_${l}_epochs${e}
            #cmd="python run_commonsense_qa_debug.py --task_name ${task} --do_eval --do_train --bert_model bert-large-uncased --data_dir /data/m1/huangl7/CommonsenseQaPlus/baselines/data/ --max_seq_length 128 --train_batch_size ${s} --learning_rate ${l} --num_train_epochs ${e} --output_dir /output/batch_${s}_lr_${l}_epochs${e} --data_dir /data/ --output_file_for_pred /output/batch_${s}_lr_${l}_epochs${e}_valid.out.jsonl"
            # ./scripts/python/cmd_with_beaker.py --gpu-count 1 --name ${name_prefix}_batch${s}_lr${l}_epoch${e}  --source ${dataset}:/data/ --cmd="${cmd}"
        done
    done
done
