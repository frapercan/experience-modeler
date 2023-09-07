import os
import json
import shutil

from experience_modeler.modeler.rollers import rollers
import yaml
import logging

from PIL import Image

import webdataset as wds
from functools import reduce

import io
import torch
from tqdm import tqdm


def decode_pt(data):
    """Decode a .pt tensor file from raw bytes."""
    stream = io.BytesIO(data)
    return torch.load(stream)


def tensor_to_bytes(tensor):
    """Convert a tensor to its binary representation."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


# read a png image


# Setting up the logger for this module
fmt = '[%(asctime)-15s] [%(levelname)s] %(name)s: %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO)
logger = logging.getLogger(__name__)


class Modeler:

    def __init__(self, config_path):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    def bulk_copy_dataset(self):
        """
        Copies the entire directory from `src` to `dest`.
        """
        src = self.config['datasets_directory']
        dest = self.config['modeled_output_directory']
        if os.path.exists(dest):
            shutil.rmtree(dest)

        try:
            shutil.copytree(src, dest)
            logger.info(f"Directory copied from {src} to {dest}")
        except shutil.Error as e:
            logger.error(f"Error: {e}")
        except OSError as e:
            logger.error(f"Error: {e}")

    def process_dataset(self):
        modeled_output_directory = self.config['modeled_output_directory']

        self.datasets = [os.path.join(modeled_output_directory, dataset_dir) for dataset_dir in
                         os.listdir(modeled_output_directory)]
        self.datasets.sort()

        for dataset in self.datasets:
            files = os.listdir(dataset)
            sample_locations = [os.path.join(dataset, file.split('.')[0]) for file in files if
                                file.split('.')[1] == 'json']
            sample_locations.sort()
            modeling_pipelines = self.config['modeling_pipelines']
            for pipeline, pipeline_conf in modeling_pipelines.items():
                params = pipeline_conf['params']
                # result = getattr(self, function)(sample_locations, **params)
                getattr(rollers, pipeline)(sample_locations, **params)

    def generate_tar_iterable(self):
        modeled_output_directory = self.config['modeled_output_directory']
        tar_output_directory = self.config['tar_output_directory']

        samples_basenames = []
        for dataset_folder in os.listdir(modeled_output_directory):
            filenames = os.listdir(os.path.join(modeled_output_directory, dataset_folder))
            basenames = [os.path.join(modeled_output_directory, dataset_folder, basename.split('.')[0]) for basename in
                         filenames if basename.split('.')[1] == 'json']
            samples_basenames.append(basenames)
        samples_basenames = list(set(reduce(lambda x, y: x + y, samples_basenames)))
        samples_basenames.sort()

        with open(self.config['metadata_output_directory'], "w") as file:
            json.dump({'length': len(samples_basenames)}, file)

        with wds.TarWriter(tar_output_directory) as sink:
            for i, basename in tqdm(enumerate(samples_basenames), total=len(samples_basenames),
                                    desc="Writing tar iterable"):

                try:
                    sample = create_sample(basename,
                                           self.config['modeling_pipelines']['rolling_img_state']['params']['horizon'])
                    # print(sample)
                    sink.write(sample)
                except Exception as e:
                    raise (e)


def create_sample(path, horizon):
    # Load the JSON data
    json_path = f"{path}.json"
    image_path = f"{path}.png"
    backward_image_paths = [f"{path}_backward_{i}.pt" for i in range(horizon)]
    forward_image_paths = [f"{path}_onward_{i}.pt" for i in range(horizon)]

    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    image = Image.open(image_path)

    # Create the sample dictionary
    basename = os.path.basename(path)  # Get the base name without the directory path
    sample = {
        "__key__": basename,  # Use the JSON file name (without extension) as the sample key
        "json": json_data,
        "image.png": image,
    }

    for i, pair in enumerate(zip(backward_image_paths, forward_image_paths)):
        backward_image_path, forward_image_path = pair[0], pair[1]
        with open(backward_image_path, 'rb') as f:
            sample[f'image_backward_{i}.pt'] = f.read()  # Read the file content as bytes
        with open(forward_image_path, 'rb') as f:
            sample[f'image_onward_{i}.pt'] = f.read()  # Read the file content as bytes

    return sample
