from _process_dataset import MyDataset, ImageTransform, make_data_path_list
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Resnet8
import torch
from torch.nn import MSELoss
from config import *
from utils import load_model
try:
  net = load_model(Resnet8(), WEIGHT_LOAD)
except:
  net = Resnet8()

# Create Dataloader
resize = IMG_SIZE_TRAIN
batch_size = BATCH_SIZE
data_root = DATA_ROOT 
list_train = make_data_path_list(root = DATA_ROOT, phase="train")
list_val = make_data_path_list(root = DATA_ROOT,phase="val")
print(f"[DATA TRAIN]:", len(list_train),"items found!")
print(f"[DATA VAL]:", len(list_val),"items found!")
train_dataset = MyDataset(list_train,classes=ACTIONS, transform=ImageTransform(resize), phase="train")
val_dataset = MyDataset(list_val,classes=ACTIONS, transform=ImageTransform(resize), phase="val")

print(train_dataset.__getitem__(0)[0].size())

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
dataloader_dict = {"train": train_dataloader, "val": val_dataloader}


# loss function***********************************
criterior = MSELoss()

# Define classes to learn
for name, param in net.named_parameters():
    param.requires_grad = True  # Always update

# Optimizer
optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)

def train(net, dataloader_dict, criterior, optimizer, epochs, save_path, save_each_epoch):
    # result
    rlt_file = open(RESULT_LOG_PATH, "w")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEVICE]: {device}")
    net.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")
        # Train
        epoch_loss = 0
        epoch_corrects =0
        min_loss = 99999
        print(dataloader_dict["train"].dataset)
        for img, lab in tqdm(dataloader_dict["train"]):
            labels, labels_oh = lab
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out = net(img)
                _, preds = torch.max(out, 1)
                loss = criterior(out, labels_oh)
                loss.backward()
                optimizer.step()
                epoch_loss+=loss.item()*img.size(0)
                epoch_corrects+= torch.sum(preds == labels.data)
        
        epoch_loss = epoch_loss/len(dataloader_dict["train"].dataset)
        epoch_accuracy = float(epoch_corrects)/len(dataloader_dict["train"].dataset)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            file_best = save_path.split(".")[0] + "_best.pth"
            torch.save(net.state_dict(), file_best)
        print(f"Train: -Loss: {epoch_loss}, - Acc: {epoch_accuracy} ")
        data_temp = f"{epoch_loss} {epoch_accuracy} "
        # Val
        epoch_loss = 0
        epoch_corrects =0
        for img, lab in tqdm(dataloader_dict["val"]):
            labels, labels_oh = lab
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                out = net(img)
                _, preds = torch.max(out, 1)
                loss = criterior(out, labels_oh)
                epoch_loss+=loss.item()*img.size(0)
                epoch_corrects+= torch.sum(preds == labels.data)
                
        
        epoch_loss = epoch_loss/len(dataloader_dict["val"].dataset)
        epoch_accuracy = float(epoch_corrects)/len(dataloader_dict["val"].dataset)

        print(f"Val: -Loss: {epoch_loss}, - Acc: {epoch_accuracy}")
        data_temp += f"{epoch_loss} {epoch_accuracy}\n"
        if save_each_epoch:
            torch.save(net.state_dict(), save_path)
        
        rlt_file.write(data_temp)
    if not save_each_epoch:
        torch.save(net.state_dict(), save_path)



print("Parameters:", sum(p.numel() for p in net.parameters()))
train(net, dataloader_dict, criterior, optimizer, EPOCHS, SAVE_PATH, SAVE_EACH_EPOCH)
