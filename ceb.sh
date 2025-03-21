python train_ceb.py \
    --wandb_key wandb.txt \
    --env Walker2d-v5 \
    --z_dim 8 \
    --beta 0.01 \
    --bs 512 \
    --n_steps 30000 \
    --model_file ./models/Walker2d-v5.pt \
    --rl_file ./policies/Walker2d-v5 \
    --ceb_file ./ceb_weights/Walker2d-v5 
    # --critic_norm \
