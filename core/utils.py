import json
import os
import uuid
import numpy as np
import torch
import cv2
from .detection.custom_config import cfg_mnet
from .detection import detec
from .recognition import rec
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
def euclidean_distance_squared(x, y):
    diff = x.unsqueeze(1) - y.unsqueeze(0)

    distances = torch.sum(diff ** 2, dim=-1)

    return distances


def load(img,net,device):

    h,w,_ =img.shape

    priordata = detec.prior_box(cfg_mnet, h, w, device)
    faces,box = detec.run(img,net,priordata,device)

    res = np.array(faces)
    src = rec.run(res)
    return src,box

def image_to_vector(img_path,detetection_net,recogition_net,device):
    print("img",img_path)
    print(os.path.exists(img_path))
    img = cv2.imread(img_path)

    h,w,_ =img.shape
    if (h > w and h > 1530):
        scale = h / 1280
        h = int(h / scale)
        w = int(w / scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    elif (w > h and w > 1530):
        scale = w / 1280
        h = int(h / scale)
        w = int(w / scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    priordata = detec.prior_box(cfg_mnet, h, w, device)
    faces,box = detec.run(img,detetection_net,priordata,device)

    res = np.array(faces)
    src = rec.faces_to_tensor(res,recogition_net)
    tensor_list = torch.unsqueeze(src[0],dim=0).tolist()
    tensor_json = json.dumps(tensor_list)
    return tensor_json

def run(img,MS,src,first_name,last_name):
    tmp_img = np.copy(img)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = detec.load_retinaface(cfg_mnet, "core/weights/mobilenet0.25_Final.pth", device)
    vectors,box = load(img,net,device)

    src =src.squeeze()
    compare =  euclidean_distance_squared(vectors,src)
    print(compare)

    argmax = torch.argmin(compare,dim=1)
    print(argmax)
    lst_attendace = []
    img_raw = np.copy(tmp_img)
    for idx, value in enumerate(argmax):
        if compare[idx][value]  < 1.2:
            lst_attendace.append(MS[value])
            name = last_name[value] + " " + first_name[value]

            b = box[idx]
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255,0), 5)
            size = (b[2] - b[0])*0.5
            cx = max(0,b[0] -(b[3]-b[1])/2)
            cy = max(0,b[1] - size)
            print("name",name)

            img_raw = Image.fromarray(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_raw)
            font = ImageFont.truetype('core/weights/times.ttf', size)
            draw.text( (cx, cy),name,font=font,fill=(255, 0, 0))
            img_raw = np.array(img_raw)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
            # cv2.putText(img_raw, name, (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255),6)
    path_save = "static/images/photo_evidence"
    jpg = str(uuid.uuid4()) + ".jpg"
    cv2.imwrite(os.path.join(path_save,jpg), img_raw)

    return lst_attendace

if __name__ == "__main__":
    image_to_vector("test.jpg",None,None,"cpu")