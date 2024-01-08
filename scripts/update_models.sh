#!/bin/bash

set -o errexit

script_dir=$(cd $(dirname $0); pwd)
cd $script_dir
model_source=/mnt/elements/models
# model_source=/home/egge/gdrive/dev/packages
cfg=${model_source}/packages-yolov4-608-3-tiny-detector.cfg
weights=${model_source}/packages-yolov4-608-3-tiny-detector_best.weights
labels=${model_source}/packages-yolov4-608-3-tiny-detector.names
onnx=/home/egge/detector/simplescan/vehicles_yolov4.onnx

# https://github.com/Tianxiaomo/pytorch-YOLOv4
source /home/egge/detector/bin/activate
PYTHONPATH=/home/egge/github/pytorch-YOLOv4
#768_1344
if [ $weights -nt $onnx ]
then
  echo "Updating $onnx"
  python3 ./darknet2onnx.py $cfg $weights $onnx
  sudo systemctl stop aicam
  #python3 ./darknet2onnx.py <(sed -e 's/^height=.*/height=608/' -e 's/^width=.*/width=608/' $cfg) $weights $onnx
  [ -e ${onnx}.engine ] && rm ${onnx}.engine
  cp -p $labels /home/egge/detector/simplescan/vehicle-labels.txt
  echo "Updated $onnx"
fi

cfg=${model_source}/ipcams-yolov4-608-3-tiny-detector.cfg
weights=${model_source}/ipcams-yolov4-608-3-tiny-detector_best.weights
labels=/${model_source}/ipcams.names
onnx=/home/egge/detector/simplescan/ipcams_color_yolov4.onnx

if [ $weights -nt $onnx ]
then
  echo "Updating $onnx"
  python3 ./darknet2onnx.py $cfg $weights $onnx
  sudo systemctl stop aicam
  #python3 ./darknet2onnx.py <(sed -e 's/^height=.*/height=768/' -e 's/^width=.*/width=1344/' $cfg) $weights $onnx
  [ -e ${onnx}.engine ] && rm ${onnx}.engine
  cp -pv $labels /home/egge/detector/simplescan/ipcams-labels.txt
  echo "Updated $onnx"
fi

cfg=${model_source}/ipcams-yolov4-608-1-tiny-detector.cfg
weights=${model_source}/ipcams-yolov4-608-1-tiny-detector_best.weights
onnx=/home/egge/detector/simplescan/ipcams_grey_yolov4.onnx
if [ $weights -nt $onnx ]
then
  echo "Updating $onnx"
  sudo systemctl stop aicam
  python3 ./darknet2onnx.py $cfg $weights $onnx
  #python3 ./darknet2onnx.py <(sed -e 's/^height=.*/height=768/' -e 's/^width=.*/width=1344/' $cfg) $weights $onnx
  [ -e ${onnx}.engine ] && rm ${onnx}.engine
  echo "Updated $onnx"
fi
sudo systemctl restart aicam
