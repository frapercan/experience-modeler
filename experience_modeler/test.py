import time

tensor_path = "/home/xaxi/PycharmProjects/experience-modeler/datasets_modeled/2048/168858279532302300/168858279760696500.pt"

import torch
import torchvision.transforms as transforms

tensor = torch.load(tensor_path)
tensor_prev = tensor[0]
tensor_ahead = tensor[1]

to_pil = transforms.ToPILImage()

for i in range(tensor_prev.shape[0]):
    image = to_pil(tensor_prev[i])
    image.show()
    time.sleep(3)


for i in range(tensor_ahead.shape[0]):
    image = to_pil(tensor_ahead[i])
    image.show()
    time.sleep(3)




