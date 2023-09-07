import json
import time

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def rolling_img_state(sample_locations, horizon,resize):
    for sample_index, sample_location in tqdm(enumerate(sample_locations),
                                              total=len(sample_locations),
                                              desc="Processing img state samples"):
        img_states = []
        clipped_horizon = horizon - 1 if sample_index > horizon else sample_index
        for sample in sample_locations[sample_index - clipped_horizon:sample_index + 1]:
            img_location = sample.split(".")[0] + '.png'
            img = Image.open(img_location)

            img_states.append(img)

        if not len(img_states):
            initial_img_state = Image.open(sample_location.split(".")[0] + '.png')
            img_states.append(initial_img_state)

        if len(img_states) >= 1 and len(img_states) < horizon:
            for i in range(horizon - len(img_states)):
                img_states.insert(0,img_states[0])

        for sample in sample_locations[sample_index + 1:sample_index + horizon + 1]:
            img_location = sample.split(".")[0] + '.png'
            img = Image.open(img_location)
            img_states.append(img)

        if len(img_states) == horizon:
            for i in range(horizon):
                img_states.append(img_states[-1])
        if len(img_states) > horizon and len(img_states) < 2 * horizon:
            filling_number = horizon - (len(img_states) - horizon)
            for i in range(filling_number):
                img_states.append(img_states[-1])

        backward_rolling = img_states[0:horizon]
        onward_rolling = img_states[horizon:]

        for i, pair_pointer_img in enumerate(zip(backward_rolling, onward_rolling)):

            transform = transforms.Compose([
                transforms.Resize((resize[0],resize[1])),
                transforms.ToTensor(),
            ])

            backward_pointer = transform(pair_pointer_img[0])
            onward_pointer = transform(pair_pointer_img[1])

            torch.save(backward_pointer,f"{sample_location}_backward_{i}.pt")
            torch.save(onward_pointer,f"{sample_location}_onward_{i}.pt")


def rolling_dimension(sample_locations, horizon, source):
    for sample_index, sample_location in tqdm(enumerate(sample_locations),
                                              total=len(sample_locations),
                                              desc="Processing numerical samples"):
        json_location = f"{sample_location}.json"
        with open(json_location, 'r+') as file:
            data = json.load(file)

            for dimension in source:
                dimension_ahead_output_name = f"rolling_{horizon}_{dimension}_ahead"
                dimension_previous_output_name = f"rolling_{horizon}_{dimension}_previous"

                data[dimension_ahead_output_name] = []
                for sample in sample_locations[sample_index + 1:sample_index + horizon + 1]:
                    with open(f"{sample}.json", 'r+') as next_file:
                        next_data = json.load(next_file)
                        data[dimension_ahead_output_name].append(next_data[dimension])

                if len(data[dimension_ahead_output_name]) == 0:
                    data[dimension_ahead_output_name].append(data[dimension])

                if len(data[dimension_ahead_output_name]) < horizon:
                    for _ in range(horizon - len(data[dimension_ahead_output_name])):
                        data[dimension_ahead_output_name].append(data[dimension_ahead_output_name][-1])


                data[dimension_previous_output_name] = []
                clipped_horizon = horizon-1 if sample_index >= horizon else sample_index
                for sample in sample_locations[sample_index - clipped_horizon:sample_index+1]:
                    with open(f"{sample}.json", 'r+') as next_file:
                        next_data = json.load(next_file)
                        data[dimension_previous_output_name].append(next_data[dimension])



                if len(data[dimension_previous_output_name]) == 0:
                    data[dimension_previous_output_name].append(data[dimension])

                if len(data[dimension_previous_output_name]) < horizon:
                    for _ in range(horizon - len(data[dimension_previous_output_name])):
                        data[dimension_previous_output_name].insert(0, data[dimension_previous_output_name][0])

            file.seek(0)
            json.dump(data, file)
