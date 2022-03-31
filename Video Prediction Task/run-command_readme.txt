CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/bair/train \
  --dataset_hparams sequence_length=17 \
  --checkpoint pretrained_models/bair_action_free/ours_savp \
  --mode test \
  --num_samples 256 \
  --results_dir results_train/bair_action_free

  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/bair/test \
  --dataset_hparams sequence_length=17 \
  --checkpoint pretrained_models/bair_action_free/ours_savp \
  --mode test \
  --num_samples 256 \
  --results_dir results_test/bair_action_free