#!/bin/bash

folder_downloads='downloads'
folder_src='src'
if [ ! -d $folder_downloads ]; then
    mkdir $folder_downloads
fi

# Download the Shapes Dataset used in "Tagger: Deep Unsupervised Perceptual Grouping"
# The url is described in https://github.com/CuriousAI/tagger/blob/master/install.sh
# 下载shapes_20x20数据集，获得文件 shapes_20x20.h5
url_shapes_20='http://cdn.cai.fi/datasets/shapes50k_20x20_compressed_v2.h5'
file_shapes_20='shapes_20x20.h5'
if [ ! -f $folder_downloads/$file_shapes_20 ]; then
    wget $url_shapes_20 -O $folder_downloads/$file_shapes_20
    # -0 infinite
fi

# Download the Static Shapes Dataset used in "Neural Expectation Maximization"
# The Dropbox url is described in https://github.com/sjoerdvansteenkiste/Neural-EM/blob/master/README.md
# 下载shapes_28x28数据集，获得文件 shapes_28x28.h5
url_shapes_28='https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AABZBL6D1KrCF8CPe-an5psoa/shapes.h5?dl=1'
file_shapes_28='shapes_28x28.h5'
if [ ! -f $folder_downloads/$file_shapes_28 ]; then
    wget $url_shapes_28 -O $folder_downloads/$file_shapes_28
fi


# Download the MNIST Dataset
# 下载MNIST数据集，获得文件 mnist.pkl.gz，并重新分组划分为train和test部分后保存为文件 MNIST.hdf5
# This python script is the modified version of https://github.com/IDSIA/brainstorm/blob/master/data/create_mnist.py
python $folder_src/download_mnist.py --folder_downloads $folder_downloads



# Convert the Shapes Dataset used in "Tagger: Deep Unsupervised Perceptual Grouping"
# 处理数据集shapes_20x20，获得文件 shapes_20x20_data.h5、shapes_20x20_labels.h5
python $folder_src/convert_multi_shapes_20x20.py --folder_downloads $folder_downloads

# Convert the Static Shapes Dataset used in "Neural Expectation Maximization"
# 处理数据集shapes_28x28，获得文件 shapes_28x28_3_data.h5、shapes_28x28_3_labels.h5
python $folder_src/convert_multi_shapes_28x28.py --folder_downloads $folder_downloads

# Create the Multi Shapes Dataset derived from "Binding via Reconstruction Clustering"
# This python script is the modified version of https://github.com/Qwlouse/Binding/blob/master/Datasets/Shapes.ipynb
# 基于数据集shapes_28x28生成包含2个和4个component的shapes数据集
# 获得文件 shapes_28x28_2_data.h5、shapes_28x28_2_labels.h5 和shapes_28x28_4_data.h5、shapes_28x28_4_labels.h5
for num_objects in 2 4; do
    python $folder_src/create_multi_shapes_28x28.py --num_objects $num_objects
done

# Create the Multi MNIST Dataset derived from "Binding via Reconstruction Clustering"
# This python script is the modified version of https://github.com/Qwlouse/Binding/blob/master/Datasets/Multi-MNIST.ipynb
# 基于数据集MNIST生成包含2个component的MNIST数据集，20 、500和all分别指什么？
# 获得文件 mnist_20_data.h5、mnist_20_labels.h5，mnist_500_data.h5 、 mnist_500_labels.h5，mnist_all_data.h5、mnist_all_labels.h5
python $folder_src/create_multi_mnist.py --folder_downloads $folder_downloads
