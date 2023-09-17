#!/bin/bash

for i in $(seq -f "%05g" 0 86)
do
    bash test.sh roofdec/data/data_real/Residential/${i}.png
done