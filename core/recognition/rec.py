import cv2
import numpy as np
import torch.nn.functional as F
from .ghostfacenetsv2 import GhostFaceNetsV2
import torch
from torchvision import transforms

trans =  transforms.Compose([
            transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output
def load_ghostfacenets():
    model = GhostFaceNetsV2(image_size=112, width=1.3, dropout=0.2, fp16=False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load("core/weights/0.9988571428571429.pt", map_location=device))
    model.eval()
    return model
def run(faces):
    model = GhostFaceNetsV2(image_size=112, width=1.3, dropout=0.2, fp16=False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load("core/weights/0.9988571428571429.pt", map_location=device))

    model.eval()
    lst = []
    flip =[]

    for face in faces:
        t =trans(face)
        lst.append(t)
        flip.append(trans(cv2.flip(face,1)))

    lst = torch.stack(lst)
    flip = torch.stack(flip)
    res = model(lst)
    res_flip = model(flip)
    res = res +res_flip
    res =l2_norm(res)
    return res

def faces_to_tensor(faces,model):
    lst = []
    flip =[]

    for face in faces:
        t =trans(face)
        lst.append(t)
        flip.append(trans(cv2.flip(face,1)))

    lst = torch.stack(lst)
    flip = torch.stack(flip)
    res = model(lst)
    res_flip = model(flip)
    res = res +res_flip
    res =l2_norm(res)
    return res
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