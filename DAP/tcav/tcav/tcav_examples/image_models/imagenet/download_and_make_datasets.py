"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
""" Downloads models and datasets for imagenet

    Content downloaded:
        - Imagenet images for the zebra class.
        - Full Broden dataset(http://netdissect.csail.mit.edu/)
        - Inception 5h model(https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception5h.py)
        - Mobilenet V2 model(https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

    Functionality:
        - Downloads open source models(Inception and Mobilenet)
        - Downloads the zebra class from imagenet, to illustrate a target class
        - Extracts three concepts from the Broden dataset(striped, dotted, zigzagged)
        - Structures the data in a format that can be readily used by TCAV
        - Creates random folders with examples from Imagenet. Those are used by TCAV.

    Example usage:

    run download_and_make_datasets.py --source_dir=./downloaded_data --number_of_images_per_folder=100 --number_of_random_folders=100
"""
import subprocess
import os
import argparse
from tensorflow.io import gfile
import imagenet_and_broden_fetcher as fetcher
import csv
import pandas as pd


def make_concepts_targets_and_randoms(source_dir, number_of_images_per_folder, number_of_random_folders):
    # Run script to download data to source_dir
    if not gfile.exists(source_dir):
        gfile.makedirs(source_dir)
    if not gfile.exists(os.path.join(source_dir,'broden1_224/')) or not gfile.exists(os.path.join(source_dir,'inception5h')):
        subprocess.call(['bash' , 'FetchDataAndModels.sh', source_dir])

    # Determine classes that we will fetch
    imagenet_labels = pd.read_csv('C:/Users/rielcheikh/Desktop/XAI/tcav/tcav/src/tcav/tcav/tcav_examples/image_models/imagenet/imagenet_url_map.csv',header=0)
    inception_labels = pd.read_csv('C:/Users/rielcheikh/Desktop/XAI/tcav/tcav/src/tcav/tcav/tcav_examples/image_models/imagenet/downloaded_data/inception5h/imagenet_comp_graph_label_strings.txt',header=0)
    imagenet = imagenet_labels['class_name'].tolist()
    inception = inception_labels['dummy'].tolist()
    
    """imagenet_classes = ['antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'persian+cat', 'horse', 
           'german+shepherd', 'blue+whale', 'siamese+cat', 'skunk', 'mole', 'hippopotamus', 'leopard', 'moose', 'spider+monkey', 
           'humpback+whale', 'elephant', 'gorilla', 'ox', 'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', 'squirrel', 'rhinoceros', 
           'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'rat', 'weasel', 'otter', 'buffalo', 'giant+panda', 'deer', 'bobcat', 'pig', 
           'mouse', 'polar+bear', 'collie', 'walrus', 'raccoon', 'dolphin']"""
    imagenet_classes = ['beaver','dalmatian','skunk','tiger','hippopotamus','leopard','gorilla','ox','chimpanzee','hamster','weasel','otter',
                        'zebra','lion','mouse','collie']
    """imagenet_classes = []
    for i in imagenet:
        for a in inception : 
            if i == a and i not in imagenet_classes: 
                imagenet_classes.append(i)"""

    """colors : black, brown, white 
    objects : fish, forest, ground, meat, tree, water
    scenes : lawn, ocean, 
    part : tail, 
    (material : water)"""
    

    broden_object_concepts = ['fish', 'forest', 'ground', 'meat', 'tree', 'water']
    broden_colors_concepts = ['black-c', 'brown-c', 'white-c']
    broden_texture_concepts = ['marbled','matted']#['blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crystalline', 'dotted',  'frilly', 'knitted', 'lacelike', 'scaly', 'striped','veined'] #['pleated','cobwebbed','grooved','bubbly','chequered','lacelike','crystalline','paisley','wrinkled','waffled','freckled','honeycombed','stratified','braided','lined','scaly','flecked','potholed','matted','cracked','studded','spiralled','swirly','zigzagged','frilly','gauzy','interlaced','grid','marbled','stained','polka-dotted','sprinkled','crosshatched','meshed','woven','perforated','veined','fibrous','pitted','knitted','porous','smeared','bumpy','striped','banded','dotted','blotchy']
    broden_scene_concepts = ['driving_range-outdoor-s',
     'reading_room-s',
     'swimming_pool-outdoor-s',
     'swimming_pool-indoor-s',
     'shopping_mall-indoor-s',
     'desert-vegetation-s',
     'clothing_store-s',
     'desert-sand-s',
     'sandbox-s',
     'sandbar-s',
     'ocean-s',
     'ice_skating_rink-indoor-s',
     'ice_skating_rink-outdoor-s',
     'mountain_snowy-s',
     'snowfield-s',
     'fire_escape-s',
     'fire_station-s',
     'moat-dry-s',
     'lake-natural-s',
     'canal-natural-s',
     'natural_history_museum-s',
     'flight_of_stairs-natural-s']
    
    ['tail', 'beak', 'head', 'ear', 'nose', 'mouth', 'hair', 'face', 'eye', 'torso', 'hand', 'arm', 'leg', 'wing', 'wheel', 'door', 
                'headlight', 'taillight', 'engine', 'text', 'horn', 'saddle', 'leaf', 'flower', 'pot', 'screen', 'skin', 'metal', 'plastic', 'wood', 
                'glass', 'leather']
    
    """broden_scene_concepts = []
    with open(source_dir+'/broden1_224/c_scene.csv') as file_obj: 
        reader_obj = csv.reader(file_obj) 
        for row in reader_obj:  
            if row[0] != 'code':
                broden_scene_concepts.append(row[2])"""

    # make targets from imagenet
    """imagenet_dataframe = fetcher.make_imagenet_dataframe("./imagenet_url_map.csv")
    for image in imagenet_classes:
        fetcher.fetch_imagenet_class(source_dir+'/awa_targets', image, number_of_images_per_folder, imagenet_dataframe)"""

    # Make concepts from broden
    for concept in broden_texture_concepts:
        fetcher.download_texture_to_working_folder(broden_path='./downloaded_data/broden1_224/',
                                                   saving_path=source_dir+'/concepts',
                                                   texture_name=concept,
                                                   number_of_images=number_of_images_per_folder)
    
    for concept in broden_scene_concepts:
       fetcher.download_scene_to_working_folder(broden_path='./downloaded_data/broden1_224/',
                                                   saving_path=source_dir+'/scenes/',
                                                   scene_name=concept,
                                                   number_of_images=number_of_images_per_folder)
       
    """for concept in broden_colors_concepts:
       fetcher.download_color_to_working_folder(broden_path='./downloaded_data/broden1_224/',
                                                   saving_path=source_dir+'/colors/',
                                                   color_name=concept,
                                                   number_of_images=number_of_images_per_folder)"""
       
    """for concept in broden_object_concepts:
       fetcher.download_object_to_working_folder(broden_path='./downloaded_data/broden1_224/',
                                                   saving_path=source_dir+'/objects/',
                                                   object_name=concept,
                                                   number_of_images=number_of_images_per_folder)"""

    # Make random folders. If we want to run N random experiments with tcav, we need N+1 folders.
    """fetcher.generate_random_folders(
        working_directory=source_dir,
        random_folder_prefix="random500",
        number_of_random_folders=number_of_random_folders+1,
        number_of_examples_per_folder=number_of_images_per_folder,
        imagenet_dataframe=imagenet_dataframe
    )"""

    

if __name__ == '__main__':
    """parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument('--source_dir', type=str,
                        help='Name for the directory where we will create the data.')
    parser.add_argument('--number_of_images_per_folder', type=int,
                        help='Number of images to be included in each folder')
    parser.add_argument('--number_of_random_folders', type=int,
                        help='Number of folders with random examples that we will generate for tcav')
    args = parser.parse_args()
    print(args.source_dir, args.number_of_images_per_folder, args.number_of_random_folders)"""
    
    # create folder if it doesn't exist
    source_dir='C:/Users/rielcheikh/Desktop/XAI/XAI-eval/data/SUN/borden_imgs' 
    number_of_images_per_folder=100 
    number_of_random_folders=100
    
    if not gfile.exists(source_dir):
        gfile.makedirs(os.path.join(source_dir))
        print("Created source directory at " + source_dir)
    # Make data
   # make_concepts_targets_and_randoms(args.source_dir, args.number_of_images_per_folder, args.number_of_random_folders)
    
    make_concepts_targets_and_randoms(source_dir, number_of_images_per_folder, number_of_random_folders)

    print("Successfully created data at " + source_dir)

