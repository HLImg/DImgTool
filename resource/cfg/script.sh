# training
accelerate launch --config_file config.yaml main.py --yaml z_local/train.yaml
# testing
python main.py --yaml z_local/test.yaml