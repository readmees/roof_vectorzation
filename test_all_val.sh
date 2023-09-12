i#!/bin/bash

for i in $(seq -f "%06g" 0 140)
do
    bash test.sh /home/ubuntu/vectorization/roof_vectorzation/data/val/${i}.jpg
done