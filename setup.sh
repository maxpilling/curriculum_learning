#!/bin/bash

current_dir=`pwd`
virtualenv -p python3 --no-site-packages --distribute .meng_env
source ${current_dir}/.meng_env/bin/activate
echo ${current_dir}/.meng_env/bin/activate
pip install -r requirements.txt
