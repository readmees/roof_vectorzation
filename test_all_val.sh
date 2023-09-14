#!/bin/bash

for i in $(seq -f "%04g" 0 17)
do
    bash test.sh data/test/${i}.png
done