# Copyright 2018 Google LLC
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

# Usage:
# bash FetchDataAndModels.sh "path_where_to_download_models_and_data"
#

# Make datasets and models
return_to="$(pwd)"
echo "Working directory is"
CUR_DIR=$(pwd)
echo $CUR_DIR

echo "folder provided is"
echo "$1"

cd "$1" # First argument contains the path

echo "starting download"



# Download mobilenet
mkdir -p mobilenet
pushd mobilenet
wget --progress=bar \
   https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz \
   -O mobilenet.zip
#curl -O https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz
#mkdir "mobilenet_v2_1.0_224"
#tar -xzvf mobilenet_v2_1.0_224.tgz -C "mobilenet_v2_1.0_224/"
#rm mobilenet_v2_1.0_224.tgz



########################## DOWNLOADING MODELS ############################
# Download inception. Model will be saved as classify_image_graph_def.pb

mkdir -p inception5h
pushd inception5h
wget --progress=bar \
   http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip \
   -O inception5h.zip

#curl -O http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
#mkdir "inception5h"
#unzip - inception5h.zip -d "inception5h/"
#rm inception5h.zip
#
#
#



######################## DOWNLOADING AND PARSING DATASETS ##################

# Download dataset concepts from CSAIL

#curl -O http://netdissect.csail.mit.edu/data/broden1_224.zip
#mkdir "broden1_224"
#unzip broden1_224.zip -d broden1_224
#rm broden1_224.zip

cd $CUR_DIR