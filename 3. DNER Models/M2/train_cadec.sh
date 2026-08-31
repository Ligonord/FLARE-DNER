
experiment_name="cadec_0001"
data_root="./data/cadec"
config_file="./training_config/cadec_working_example.jsonnet"
cuda_device=$1

allennlp train $config_file \
    --cache-directory $data_root/cached \
    --serialization-dir ./models/$experiment_name \
    --include-package sodner \
    -f