# Object Detection API with TensorFlow 2

## Requirements

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Protobuf Compiler >= 3.0](https://img.shields.io/badge/ProtoBuf%20Compiler-%3E3.0-brightgreen)](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager)

## Installation

You can install the TensorFlow Object Detection API either with Python Package
Installer (pip) or Docker. For local runs we recommend using Docker and for
Google Cloud runs we recommend using pip.

Clone the TensorFlow Models repository and proceed to one of the installation
options.

```bash
git clone https://github.com/tensorflow/models.git
```

### Docker Installation

```bash
# From the root of the git repository
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t od .
docker run -it od
```

### Python Package Installation

```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

For installation with TF2.3 ( works better with tf2onnx 1.9) 
```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup_2.3.py setup.py
python -m pip install --use-feature=2020-resolver .
```


```bash
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```
Some tests may fail with TF2.3 based ( which is expected).

## Quick Start

### Colabs

<!-- mdlint off(URL_BAD_G3DOC_PATH) -->

*   Training -
    [Fine-tune a pre-trained detector in eager mode on custom data](../colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb)

*   Inference -
    [Run inference with models from the zoo](../colab_tutorials/inference_tf2_colab.ipynb)

*   Few Shot Learning for Mobile Inference -
    [Fine-tune a pre-trained detector for use with TensorFlow Lite](../colab_tutorials/eager_few_shot_od_training_tflite.ipynb)

<!-- mdlint on -->

## Training and Evaluation

To train and evaluate your models either locally or on Google Cloud see
[instructions](tf2_training_and_evaluation.md).

## Model Zoo

We provide a large collection of models that are trained on COCO 2017 in the
[Model Zoo](tf2_detection_zoo.md).

## Model Export (Fixed input resolution)
Use following command to export the models without pre-processing assuming fixed size of input-image width and height 

```bash
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path <path to pipeline.config> --trained_checkpoint_dir <path to checkpoint> --output_directory <path where saved_model will be created> -skip_preprocess -input_dims <fixed input width>,<fixed input height>
```

## Guides

*   <a href='configuring_jobs.md'>
      Configuring an object detection pipeline</a><br>
*   <a href='preparing_inputs.md'>Preparing inputs</a><br>
*   <a href='defining_your_own_model.md'>
      Defining your own model architecture</a><br>
*   <a href='using_your_own_dataset.md'>
      Bringing in your own dataset</a><br>
*   <a href='evaluation_protocols.md'>
      Supported object detection evaluation protocols</a><br>
*   <a href='tpu_compatibility.md'>
      TPU compatible detection pipelines</a><br>
*   <a href='tf2_training_and_evaluation.md'>
      Training and evaluation guide (CPU, GPU, or TPU)</a><br>
