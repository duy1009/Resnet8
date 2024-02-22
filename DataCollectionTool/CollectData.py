import sys
sys.path.insert(0, './Resnet8/DataCollectionTool')
import airsim
import time
from pynput import keyboard
from config import *
import cv2
import sys
from math import pi, cos, sin
import os
from utils import *

from pathlib import Path
ROOT = Path(__file__).parent

for i in ACTIONS:
    make_folder(f"{ROOT}\\Data\\{i}")


f_name = get_name_f(f"{ROOT}\\Data\\images", "data", START_FILE_NAME)
print(f_name)


save = quit = False
cnt = 0
def UAVStart():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    client.enableApiControl(True)
    print("arming the drone...")
    client.armDisarm(True)

    landed = client.getMultirotorState().landed_state
    if landed == airsim.LandedState.Landed:
        print("taking off...")
        client.takeoffAsync().join()

    landed = client.getMultirotorState().landed_state
    if landed == airsim.LandedState.Landed:
        print("takeoff failed - check Unreal message log for details")
        exit()
    return client
client = UAVStart()

def on_key_press(key):
    global save, quit, cnt, client

    x_temp, y_temp = 0, 0
    z_val = 0
    action = ACTIONS["IDLE"]
    #  Get data

    x,y,z=client.simGetVehiclePose().position
    print(x,y,z)
    pitch, roll, yaw = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
    yaw_deg = yaw*180/pi
    if key == keyboard.KeyCode.from_char("p"):
        x_temp, y_temp = 0, 0
        action = "IDLE"
    if key == keyboard.KeyCode.from_char("w"):
        x_temp, y_temp = 1.5*UAV_VELOCITY, 0
        action = "FRONT"
    if key == keyboard.KeyCode.from_char("s"):
        action = "BACK"
        x_temp, y_temp = -1.5*UAV_VELOCITY, 0
    if key == keyboard.KeyCode.from_char("a"):
        action = "LEFT"
        yaw_deg -= 90
        x_temp, y_temp =  0, -1.5*UAV_VELOCITY
    if key == keyboard.KeyCode.from_char("d"):
        action = "RIGHT"
        yaw_deg += 90
        x_temp, y_temp =  0, -1.5*UAV_VELOCITY
    if key == keyboard.KeyCode.from_char("q"):
        action = "FRONT_LEFT"
        yaw_deg -= TURN_ANGLE*180/pi 
        x_temp, y_temp =  1.5*UAV_VELOCITY* cos(TURN_ANGLE), -1.5*UAV_VELOCITY* sin(TURN_ANGLE)
    if key == keyboard.KeyCode.from_char("e"):
        action = "FRONT_RIGHT"
        yaw_deg += TURN_ANGLE*180/pi 
        x_temp, y_temp =  1.5*UAV_VELOCITY* cos(TURN_ANGLE), 1.5*UAV_VELOCITY* sin(TURN_ANGLE)
    
    if key == keyboard.KeyCode.from_char("t"):
        z_val+= UAV_VELOCITY*1.5
    if key == keyboard.KeyCode.from_char("y"):
        z_val-= UAV_VELOCITY*1.5


    x_val, y_val  = rollPoint2D((x_temp, y_temp), yaw)
    
    if save:
        path_save = f"{ROOT}\\Data\\{action}\\{f_name}_{cnt}.png" 
        img = getRGBImg(client)
        cv2.imwrite(path_save, img)
        cnt+=1
    st = time.time()
    try:
        client.moveToPositionAsync(x+x_val, y+y_val,z+z_val, UAV_VELOCITY,
                                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                        yaw_mode=airsim.YawMode(False, yaw_deg))
    except:
        print("ERROR")
    if key == keyboard.KeyCode.from_char("7"):
        save = True
        print("[Start]")
    if key == keyboard.KeyCode.from_char("8"):
        save = False
        print("[Stop]")

    if key == keyboard.Key.esc:
        quit = True


def getRGBImg(client):
        rawImage = client.simGetImage("0", airsim.ImageType.Scene)
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(png, cv2.COLOR_RGBA2RGB)
            return img


with keyboard.Listener(on_press=on_key_press) as listener:
    listener.join()
    while True:
        
        if quit:
            print("Stop")
            break
        time.sleep(0.01)

    listener.stop()
    
    # cnt = 0
    # pre_time = 0
    # while True:
    #     x_val, y_val, z_val = 0, 0, 0
    #     #  Get data
    #     x,y,z=client.simGetVehiclePose().position
    #     pitch, roll, yaw = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
    #     img = getRGBImg(client)
        
        
            
    #     time.sleep(0.1)
            

