from math import acos, pi, sqrt, cos, sin
import os
import glob
def angle(vector):
    x, y = vector[0],vector[1]
    if y>=0:
        return acos(x/sqrt(x**2+y**2))
    else:
        return 2*pi - acos(x/sqrt(x**2+y**2))
def distance3D(vector):
    (x,y,z) = vector
    return sqrt(x**2+y**2+z**2)

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
                temp.append(j)
            datas.append([temp])
    return datas

def make_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def get_name_f(path, name, st_fn = 0):
    max = st_fn
    for i in glob.glob(path+"\\*"):
        try: 
            num = int(i.split("\\")[-1].split("_")[1])
            if num>max:    
                max = num
        except: continue
    return f"{name}_{max+1}"
# client.hoverAsync().join()  
        # client.landAsync().join()  
        # print("landed") 

def rollPoint2D(point, angle):
    x,y = point
    return x*cos(angle)-y*sin(angle), x*sin(angle) + y*cos(angle)