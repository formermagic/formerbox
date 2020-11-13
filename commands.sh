formerbox-cli train_tokenizer  \
    --tokenizer code-roberta  \
    --save_directory /Users/mozharovsky/Desktop/gitnetic-datasets/tiny_dataset/tokenizer \
    --files /Users/mozharovsky/GitHub/gitnetic-ml/tests/fixtures/tiny_dataset/tiny_dataset.train.src \
    --vocab_size 20000 \
    --legacy_format false


formerbox-cli preprocess  \
    --train_prefix /Users/mozharovsky/GitHub/gitnetic-ml/tests/fixtures/tiny_dataset/tiny_dataset.train.src  \
    --valid_prefix /Users/mozharovsky/GitHub/gitnetic-ml/tests/fixtures/tiny_dataset/tiny_dataset.valid.src  \
    --test_prefix /Users/mozharovsky/GitHub/gitnetic-ml/tests/fixtures/tiny_dataset/tiny_dataset.test.src  \
    --tokenizer roberta \
    --tokenizer_path /Users/mozharovsky/Desktop/gitnetic-datasets/tiny_dataset/tokenizer  \
    --binarizer transformer-binarizer  \
    --max_length 512  \
    --return_overflowing_tokens true  \
    --output_path /Users/mozharovsky/Desktop/gitnetic-datasets/tiny_dataset/new_dataset  \
    --num_proc 8  \
    --batch_size 512  \
    --batched true  \
    --dataset_impl mmap --legacy_format false

formerbox-cli train --task transformer-task \
  --config_path /Users/mozharovsky/GitHub/gitnetic-ml/bart-config.yml \
  --tokenizer_path /Users/mozharovsky/Desktop/gitnetic-datasets/tiny_dataset/tokenizer \
  --warmup_steps 4000 \
  --learning_rate 5e-5 \
  --power 1.0 \
  --train_data_prefix /Users/mozharovsky/Desktop/gitnetic-datasets/tiny_dataset/new_dataset/tiny_dataset.train.src \
  --val_data_prefix /Users/mozharovsky/Desktop/gitnetic-datasets/tiny_dataset/new_dataset/tiny_dataset.valid.src \
  --max_tokens 3072 \
  --num_workers 32 \
  --max_steps 1500000 \
  --val_check_interval 50 \
  --save_step_frequency 3000 \
  --num_sanity_val_steps 0 \
  --save_dir save_dir \
  --progress_bar_refresh_rate 1 \
  --log_every_n_steps 1 \
  --gpus 0 \
  --resume_from_checkpoint /Users/mozharovsky/Desktop/checkpoint_epoch=2_global_step=387000.ckpt