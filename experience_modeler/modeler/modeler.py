import os
import json
import shutil

import torch
import yaml
import logging

from PIL import Image
# import torch
import torchvision
from torchvision import transforms

# read a png image


# Setting up the logger for this module
fmt = '[%(asctime)-15s] [%(levelname)s] %(name)s: %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO)
logger = logging.getLogger(__name__)


class Modeler:

    def __init__(self, config_path):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)


    def rolling(self, sample_locations, horizon, source):
        non_state_dimensions = [source_i for source_i in source if source_i != 'state']
        for sample_index, sample_location in enumerate(sample_locations):
            # Numerical information
            with open(sample_location, 'r+') as file:
                data = json.load(file)


                for dimension in non_state_dimensions:
                    dimension_output_name = f"rolling_{horizon}_{dimension}"
                    data[dimension_output_name] = []
                    data[dimension_output_name].append(data[dimension])
                    for sample in sample_locations[sample_index+1:sample_index + horizon]:
                        with open(sample, 'r+') as next_file:
                            next_data = json.load(next_file)
                            data[dimension_output_name].append(next_data[dimension])

                file.seek(0)
                json.dump(data, file)



            if 'state' in source:
                tensor_list = []
                for sample in sample_locations[sample_index:sample_index + horizon]:
                    img_location = sample.split(".")[0] + '.png'

                    print(img_location)
                    # Load the image using PIL
                    img = Image.open(img_location)

                    # Define the transformation to convert the image to a tensor
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])

                    # Convert the image to a tensor
                    tensor = transform(img)

                    # Append tensor to the list
                    tensor_list.append(tensor)

                # Stack rolling tensors
                tensor_location = f"{sample_location.split('.')[0]}_state_{horizon}.pt"
                stacked_tensors = torch.stack(tensor_list, dim=0)
                torch.save(stacked_tensors, tensor_location)


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

