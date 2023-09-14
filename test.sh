# The next part is to make an edge image to put into the neural network using HED
fn=$1
# Split the string by the last dot (.)
base_name="${fn%.*}"
extension="${fn##*.}"

# Create the new string with "_edge" and "_enhanced" added
edge_fn="${base_name}_edge.${extension}"
enhance_fn="${base_name}_enhanced.${extension}"

# Detect edges with HED
python pytorch-hed/run.py --in $fn --out $edge_fn
echo "Original: $fn, created from edge: $edge_fn"

# Draw the edges on image
python edge_enhancer.py --original=$fn --edge=$edge_fn --output=$enhance_fn
echo "Enhanced: $enhance_fn, created from edge: $edge_fn, and original: $fn"

# Use HED-trained model on HED data
python test.py \
--exp-name line_weighted_wo_focal_junc --backbone resnet50 \
--backbone-kwargs '{"encoder_weights": "ckpt/backbone/encoder_epoch_20.pth", "decoder_weights": "ckpt/backbone/decoder_epoch_20.pth"}' \
--dim-embedding 256 --junction-pooling-threshold 0.3 \
--junc-pooling-size 64 --block-inference-size 128 \
--gpus 0, --resume-epoch latest \
--vis-junc-th 0.25 --vis-line-th 0.25 \
    - test $enhance_fn
