#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/coco
cd data/coco

wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -o val2017.zip
unzip -o annotations_trainval2017.zip

test -d val2017
test -f annotations/instances_val2017.json

echo "COCO val2017 ready"
