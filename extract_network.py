import PIL
import PIL.Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

from models.percept_net import PerceptNet
from models.planner_net import PlannerNet

OUTFOLDER = "output"
TEST_IMAGE = "test_image.png"
# SD = State dict so we don't have to have whole iPlanner module loaded which introduces a bunch of dependencies
FULL_NETWORK_SD = "trained_weights/plannernet_robot_dimension_emb256_low_resolution_SD.pt"
ENCODER_SD_SAVE_LOCATION = "extracted_weights/perceptnet_emb256_low_resolution_SD.pt"

def plotAndSave(img, filename = "image", path = ""):
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path,filename+".png"))
    plt.close()

# load full network
net = PlannerNet(encoder_channel=16, k=5)
net.load_state_dict(torch.load(FULL_NETWORK_SD))
net = net.to('cuda:0')

# extract and save
encoder = net.encoder
torch.save(encoder.state_dict(), ENCODER_SD_SAVE_LOCATION)

# load perception part
encoder_net = PerceptNet(layers=[2, 2, 2, 2])
encoder_net.load_state_dict(torch.load(ENCODER_SD_SAVE_LOCATION))
encoder_net = encoder_net.to('cuda:0')


# load image in same way as during training
depth_transform = transforms.Compose([
    transforms.Resize(tuple([180, 320])),
    transforms.ToTensor()])

image = PIL.Image.open(TEST_IMAGE)
image = np.array(image)
plotAndSave(image, "1", OUTFOLDER)

image[~np.isfinite(image)] = 0.0
plotAndSave(image, "2", OUTFOLDER)

image = (image / 1000.0).astype("float32")
plotAndSave(image, "3", OUTFOLDER)

image[image > 10] = 0.0
plotAndSave(image, "4", OUTFOLDER)

img = PIL.Image.fromarray(image)
plotAndSave(img, "5", OUTFOLDER)

img = depth_transform(img).expand(1, 3, -1, -1)
plotAndSave(img[0,0,...], "6", OUTFOLDER)

img = img.to('cuda:0')

with torch.no_grad():
    output = encoder_net(img)
    print(output.shape)

