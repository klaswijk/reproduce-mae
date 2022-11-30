#!/bin/bash
# $1 = version of COCO, e.g. 2014, 2017
trap "exit" INT
mkdir -p ./data
mkdir -p ./data/coco

train=http://images.cocodataset.org/zips/train$1.zip
val=http://images.cocodataset.org/zips/val$1.zip
ann=http://images.cocodataset.org/annotations/annotations_trainval$1.zip

if command -v aria2c > /dev/null 2>&1; then
    aria2c -x 10 -j 10 -d ./data/coco --continue=true $train
    aria2c -x 10 -j 10 -d ./data/coco --continue=true $val
    aria2c -x 10 -j 10 -d ./data/coco --continue=true $ann
else
    echo "aria2c not installed, using wget"
    wget -d -nc ./data/coco $train
    wget -d -nc ./data/coco $val
    wget -d -nc ./data/coco $ann
fi

n_files=`unzip -l ./data/coco/train$1.zip | tail -n 1 | xargs echo -n | cut -d' ' -f2`
unzip -n ./data/coco/train$1.zip -d ./data/coco/train$1 \
    | tqdm --desc extracted --unit files --unit_scale --total $n_files > /dev/null

n_files=`unzip -l ./data/coco/val$1.zip \
    | tail -n 1 | xargs echo -n | cut -d' ' -f2`
unzip -n ./data/coco/val$1.zip -d ./data/coco/val$1 \
    | tqdm --desc extracted --unit files --unit_scale --total $n_files > /dev/null

n_files=`unzip -l ./data/coco/annotations_trainval$1.zip | tail -n 1 | xargs echo -n | cut -d' ' -f2`
unzip -n ./data/coco/annotations_trainval$1.zip -d ./data/coco/annotations_trainval$1 \
    | tqdm --desc extracted --unit files --unit_scale --total $n_files > /dev/null