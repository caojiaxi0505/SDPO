cd /cfs_turbo/jiaxicao/OPSD
export TENSORBOARD_DIR=/cfs_turbo/jiaxicao/tensorboard/grpo-tooluse-qwen3-8b-h20
export ROLLOUT_DIR=/cfs_turbo/jiaxicao/rollouts/grpo-tooluse-qwen3-8b-h20
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash training/verl_training.sh \
  grpo-tooluse-qwen3-8b-h20 \
  baseline_grpo \
  datasets/tooluse \
  vars.dir=/cfs_turbo/jiaxicao/OPSD \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.model.path=/cfs_turbo/jiaxicao/ckpt/hf/Qwen3-8B \
  critic.model.path=/cfs_turbo/jiaxicao/ckpt/hf/Qwen3-8B \
  data.train_batch_size=32 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  data.max_response_length=8192 \
  max_model_len=10240 \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  trainer.total_epochs=5 \
  trainer.test_freq=5 \
  trainer.save_freq=1000000 \
  trainer.default_local_dir=/cfs_turbo/jiaxicao/ckpt/ucsdpo_runs/grpo-tooluse-qwen3-8b-h20 \
  trainer.rollout_data_dir=${ROLLOUT_DIR}/train \
  trainer.validation_data_dir=${ROLLOUT_DIR}/val \
  'trainer.logger=["console","tensorboard"]'
