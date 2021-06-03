#!/bin/bash

set -o errexit

script_dir=$(cd $(dirname $0); pwd)
cd $script_dir
cfg=/home/egge/gdrive/dev/packages/packages-yolov4-608-3-tiny-detector.cfg
weights=/home/egge/gdrive/dev/packages/packages-yolov4-608-3-tiny-detector_best.weights
labels=/home/egge/gdrive/dev/packages/obj.names
onnx=/home/egge/detector/simplescan/vehicles_yolov4.onnx

PYTHONPATH=/home/egge/github/pytorch-YOLOv4
#768_1344
if [ $weights -nt $onnx ]
then
  echo "Updating $onnx"
  python3 ./darknet2onnx.py $cfg $weights $onnx
  #python3 ./darknet2onnx.py <(sed -e 's/^height=.*/height=608/' -e 's/^width=.*/width=608/' $cfg) $weights $onnx
  [ -e ${onnx}.engine ] && rm ${onnx}.engine
  cp -p $labels /home/egge/detector/simplescan/vehicle-labels.txt
  echo "Updated $onnx"
fi

cfg=/home/egge/gdrive/dev/ipcams/ipcams-yolov4-608-3-tiny-detector.cfg
weights=/home/egge/gdrive/dev/ipcams/ipcams-yolov4-608-3-tiny-detector_best.weights
labels=/home/egge/gdrive/dev/ipcams/obj.names
onnx=/home/egge/detector/simplescan/ipcams_color_yolov4.onnx

if [ $weights -nt $onnx ]
then
  echo "Updating $onnx"
  python3 ./darknet2onnx.py $cfg $weights $onnx
  #python3 ./darknet2onnx.py <(sed -e 's/^height=.*/height=768/' -e 's/^width=.*/width=1344/' $cfg) $weights $onnx
  [ -e ${onnx}.engine ] && rm ${onnx}.engine
  cp -pv $labels /home/egge/detector/simplescan/ipcams-labels.txt
  echo "Updated $onnx"
fi

cfg=/home/egge/gdrive/dev/ipcams/ipcams-yolov4-608-1-tiny-detector.cfg
weights=/home/egge/gdrive/dev/ipcams/ipcams-yolov4-608-1-tiny-detector_best.weights
onnx=/home/egge/detector/simplescan/ipcams_grey_yolov4.onnx
if [ $weights -nt $onnx ]
then
  echo "Updating $onnx"
  python3 ./darknet2onnx.py $cfg $weights $onnx
  #python3 ./darknet2onnx.py <(sed -e 's/^height=.*/height=768/' -e 's/^width=.*/width=1344/' $cfg) $weights $onnx
  [ -e ${onnx}.engine ] && rm ${onnx}.engine
  echo "Updated $onnx"
fi
echo sudo systemctl restart aicam
