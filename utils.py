
import os
import cv2
import sys
import torch

def isCorrect(vectorPred, vecLabel, error_val):
    return error_val > (vecLabel - vectorPred).pow(2).sum().sqrt().item()

def numOfCorrect(vectorPreds, vecLabels, error_val):
    cnt = 0
    for i in range(vectorPreds.size()[0]):
        if isCorrect(vectorPreds[i], vecLabels[i], error_val):
            cnt+=1
    return cnt


def processListPath(list_path):
    lis = []
    for i in list_path:
        i=i.replace("\\", "/")
        lis.append(i)
    return lis
def writeTxtData(path, datas):
    data = ""
    for i in datas:
        for j in i:
            data+= f"{j} "
        data+="\n"
    f = open(path, "w")
    f.write(data)
    f.close()

def loadEnvData(path):
    datas = []
    if os.path.isfile(path):
        f=open(path, "r")
        data = f.read().split("\n")[:-1]
        f.close()
        for i in data:
            values = i.split(" ")
            temp = []
            for j in values:
                try:
                    temp.append(float(j))
                except: continue
            datas.append(temp)
    return datas



def getRGBImg(client):
        rawImage = client.simGetImage("0", airsim.ImageType.Scene)
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(png, cv2.COLOR_RGBA2RGB)
            return img

def getPILImg(client):
        rawImage = client.simGetImage("0", airsim.ImageType.Scene)
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(png, cv2.COLOR_BGR2RGB)
            return img

def load_model(net, path="./weight.pth"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    load_weight = torch.load(path, map_location=device)
    
    print(f"[DEVICE]: {device}")
    # load_weight = torch.load(path, map_location=("cuda:0"))
    net.load_state_dict(load_weight)
    return net