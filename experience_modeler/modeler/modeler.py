import io
import os
import json
import shutil
import time

import numpy as np
import yaml
import logging

from PIL import Image
from torchvision import transforms

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
            self.transform = transforms.Compose([
                transforms.Resize((self.config['resampling_scale'][0], self.config['resampling_scale'][1])),
                transforms.ToTensor(),
            ])

    def rolling(self, sample_locations, horizon, source):
        non_state_dimensions = [source_i for source_i in source if source_i != 'state']
        with open(self.config['metadata_output_directory'], "w") as file:
            json.dump({'length': len(sample_locations)}, file)
        for sample_index, sample_location in tqdm(enumerate(sample_locations), total=len(sample_locations),
                                                  desc="Processing samples"):
            tensor_location = f"{sample_location.split('.')[0]}.pt"

            # Numerical information
            with open(sample_location, 'r+') as file:
                data = json.load(file)

                for dimension in non_state_dimensions:
                    dimension_ahead_output_name = f"rolling_{horizon}_{dimension}_ahead"
                    dimension_previous_output_name = f"rolling_{horizon}_{dimension}_previous"
                    data[dimension_previous_output_name] = []
                    data[dimension_ahead_output_name] = []
                    for sample in sample_locations[sample_index+1:sample_index + horizon+1]:
                        with open(sample, 'r+') as next_file:
                            next_data = json.load(next_file)
                            data[dimension_ahead_output_name].append(next_data[dimension])

                    clipped_horizon = horizon if sample_index > horizon else sample_index
                    for sample in sample_locations[sample_index - clipped_horizon:sample_index+1]:
                        with open(sample, 'r+') as next_file:
                            next_data = json.load(next_file)
                            data[dimension_previous_output_name].append(next_data[dimension])

                file.seek(0)
                json.dump(data, file)

            if 'state' in source:
                ## Previous
                previous_states = []
                clipped_horizon = horizon if sample_index > horizon else sample_index
                for sample in sample_locations[sample_index - clipped_horizon:sample_index]:
                    img_location = sample.split(".")[0] + '.png'
                    # Load the image using PIL
                    img = Image.open(img_location)

                    # Append tensor to the list
                    previous_states.append(img)

                # Stack rolling tensors

                if not len(previous_states):
                    img_location = sample_location.split(".")[0] + '.png'
                    initial_sample = Image.open(img_location)
                    previous_states.append(initial_sample)

                if len(previous_states) >= 1 and len(previous_states) < horizon:

                    for i in range(horizon - len(previous_states)):
                        previous_states.insert(0,previous_states[0])

                for i,previous_state in enumerate(previous_states):
                    new_location = f"{sample_location.split('.')[0]}_previous_state_{horizon-i}.png"
                    previous_state.save(new_location)


                ## AHEAD
                ahead_states = []
                for sample in sample_locations[sample_index + 1:sample_index + horizon + 1]:
                    img_location = sample.split(".")[0] + '.png'

                    # Load the image using PIL
                    img = Image.open(img_location)

                    # Convert the image to a tensor

                    # Append tensor to the list
                    ahead_states.append(img)

                # Stack rolling tensors
                if len(ahead_states) == 0:
                    for i in range(horizon):
                        ahead_states.append(previous_states[-1])

                if len(ahead_states) > 0 and len(ahead_states) < horizon:
                    filling_number = horizon - len(ahead_states)
                    for i in range(filling_number):
                        ahead_states.append(previous_states[-1])



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
            sample_locations = [os.path.join(dataset, file) for file in files if file.split('.')[1] == 'json']
            sample_locations.sort()
            modeling_pipelines = self.config['modeling_pipelines']
            for pipeline, pipeline_conf in modeling_pipelines.items():
                for function, function_conf in modeling_pipelines[pipeline]['functions'].items():
                    params = function_conf['params']
                    result = getattr(self, function)(sample_locations, **params)

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

        with wds.TarWriter(tar_output_directory) as sink:
            for i, basename in tqdm(enumerate(samples_basenames), total=len(samples_basenames),
                                    desc="Writing tar iterable"):

                try:
                    sample = create_sample(basename)
                    # print(sample)
                    sink.write(sample)
                except Exception as e:
                    raise (e)


#
#
#
def create_sample(path):
    # Load the JSON data
    json_path = f"{path}.json"
    image_path = f"{path}.png"
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

    return sample
