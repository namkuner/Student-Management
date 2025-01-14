import cv2
import torch.nn.functional as F
from ghostfacenetsv2 import GhostFaceNetsV2
import torch
from torchvision import transforms
model = GhostFaceNetsV2(image_size=112, width=1.3)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load("weights/0.9988571428571429.pt",map_location=device))

model.eval()
trans =  transforms.Compose([
            transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

# img1 = cv2.imread("images/003475.jpg")
# img1 =cv2.resize(img1,(112,112))
# img1 = trans(img1)
# print(img1.shape)
# img1 = torch.unsqueeze(img1, dim=0)
# img1 = l2_norm(img1)
# print(img1.shape)
# res = model(img1)
#
# print(res.shape)
