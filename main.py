from detection.custom_config import cfg_mnet
from detection import detect
import cv2
import torch
from spoofing.spoof import load_weight

if __name__ =="__main__":


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vid = cv2.VideoCapture(0)

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    priordata = detect.prior_box(cfg_mnet, 720, 1280,device)
    net = detect.load_retinaface(cfg_mnet, "weights/mobilenet0.25_Final.pth", device)
    spoof_model = load_weight()
    spoof_model.eval()

    while (True):

        ret, frame = vid.read()

        if ret:
            try:
                frame = cv2.flip(frame,1)
                res = detect.run_spoofing(frame, net, priordata, spoof_model, device)
                cv2.imshow('frame', res)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(e)

    vid.release()
    cv2.destroyAllWindows()
