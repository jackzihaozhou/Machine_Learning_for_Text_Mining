#!/bin/bash

echo "Running using train file at" $1 "and test file at" $2

python3 hvm_newton.py $1 $2
python3 hvm_sgd.py $1 $2

# this bash will create a directory at current path name "result" and store the evaluate result
