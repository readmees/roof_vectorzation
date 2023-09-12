python test.py \
--exp-name line_weighted_wo_focal_junc --backbone resnet50 \
--backbone-kwargs '{"encoder_weights": "ckpt/backbone/encoder_epoch_20.pth", "decoder_weights": "ckpt/backbone/decoder_epoch_20.pth"}' \
--dim-embedding 256 --junction-pooling-threshold 0.3 \
--junc-pooling-size 64 --block-inference-size 128 \
--gpus 0, --resume-epoch latest \
--vis-junc-th 0.25 --vis-line-th 0.25 \
    - test $1
