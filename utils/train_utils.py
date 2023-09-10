from tqdm import tqdm
import sys
import torch.nn as nn
import torch
import numpy as np

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def train_one_epoch(model,optimizer,data_loader,epoch,device,scaler):
    model.train()
    criterion = nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout)

    cost_vector = []
    class_cost_vector = []
    domain_cost_vector = []
    acc_vector = []
    for step, (train_data, train_labels,event_labels) in enumerate(data_loader):
        train_text, train_image, train_mask, train_labels,event_labels = \
            train_data[0].to(device), train_data[1].to(device),train_data[2].to(device),train_labels.to(device),event_labels.to(device),

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            class_outputs, domain_outputs = model(train_text, train_image,train_mask)

            ## Fake or Real loss
            class_loss = criterion(class_outputs, train_labels)
            # Event Loss
            domain_loss = criterion(domain_outputs, event_labels)
            loss = class_loss + domain_loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch+1, round(mean_loss.item(), 3))

        _, argmax = torch.max(class_outputs, 1)
        accuracy = (train_labels == argmax.squeeze()).float().mean()
        class_cost_vector.append(class_loss.item())
        domain_cost_vector.append(domain_loss.item())
        cost_vector.append(loss.item())
        acc_vector.append(accuracy.item())

    return cost_vector,class_cost_vector,domain_cost_vector,acc_vector

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    validate_acc_vector = []
    criterion = nn.CrossEntropyLoss()
    data_loader = tqdm(data_loader, file=sys.stdout)
    valid_cost_vector = []
    for i, (validate_data, validate_labels,event_labels) in enumerate(data_loader):
        validate_text, validate_image, validate_mask, validate_labels,event_labels = \
            validate_data[0].to(device), validate_data[1].to(device), validate_data[2].to(device),\
            validate_labels.to(device),event_labels.to(device)
        validate_outputs, domain_outputs = model(validate_text, validate_image,validate_mask)
        _, validate_argmax = torch.max(validate_outputs, 1)
        valid_loss = criterion(validate_outputs, validate_labels)
        valid_cost_vector.append(valid_loss.item())
        # domain_loss = criterion(domain_outputs, event_labels)
        validate_accuracy = (validate_labels == validate_argmax).float().sum()
        validate_acc_vector.append(validate_accuracy.item())
        if i == 0:
            valid_pred = to_np(validate_argmax) #tensor -> numpy
            valid_true = to_np(validate_labels)
        else:
            valid_pred = np.concatenate((valid_pred,to_np(validate_argmax)),axis=0)
            valid_true = np.concatenate((valid_true,to_np(validate_labels)),axis=0)

    return validate_acc_vector,valid_cost_vector,valid_pred,valid_true

def train_model_one_epoch_without_event(model,optimizer,data_loader,epoch,device,scaler):
    model.train()
    criterion = nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout)

    cost_vector = []
    class_cost_vector = []
    domain_cost_vector = []
    acc_vector = []

    for step, (train_data, train_labels) in enumerate(data_loader):
        train_text, train_image, train_mask, train_labels= \
            train_data[0].to(device), train_data[1].to(device),train_data[2].to(device),train_labels.to(device),

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            class_outputs,_ = model(train_text, train_image, train_mask) # EANN model has two losses
            # class_outputs= model(train_text, train_image, train_mask) # MCAN model has one loss
            ## Fake or Real loss
            class_loss = criterion(class_outputs, train_labels)
            # Event Loss
            # domain_loss = criterion(domain_outputs, event_labels)
            # domain_loss = torch.zeros(1).to(device)
            loss = class_loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch+1, round(mean_loss.item(), 3))

        _, argmax = torch.max(class_outputs, 1)
        accuracy = (train_labels == argmax.squeeze()).float().mean()
        class_cost_vector.append(class_loss.item())
        # domain_cost_vector.append(domain_loss.item())
        cost_vector.append(loss.item())
        acc_vector.append(accuracy.item())

    return cost_vector,class_cost_vector,acc_vector

@torch.no_grad()
def evaluate_model_without_event(model, data_loader, device):
    model.eval()
    validate_acc_vector = []
    criterion = nn.CrossEntropyLoss()
    data_loader = tqdm(data_loader, file=sys.stdout)
    valid_cost_vector = []
    for i, (validate_data, validate_labels) in enumerate(data_loader):
        validate_text, validate_image, validate_mask, validate_labels = \
            validate_data[0].to(device), validate_data[1].to(device), validate_data[2].to(device),\
            validate_labels.to(device)
        validate_outputs,_ = model(validate_text, validate_image,validate_mask) # EANN model has two losses
        # validate_outputs = model(validate_text, validate_image, validate_mask) # MCAN model has one loss
        _, validate_argmax = torch.max(validate_outputs, 1)
        valid_loss = criterion(validate_outputs, validate_labels)
        valid_cost_vector.append(valid_loss.item())
        validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().sum()
        validate_acc_vector.append(validate_accuracy.item())
        if i == 0:
            valid_pred = to_np(validate_argmax) #tensor -> numpy
            valid_true = to_np(validate_labels)
        else:
            valid_pred = np.concatenate((valid_pred,to_np(validate_argmax)),axis=0)
            valid_true = np.concatenate((valid_true,to_np(validate_labels)),axis=0)

    return validate_acc_vector,valid_cost_vector,valid_pred,valid_true


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def reset(self):
        if self.matrix is not None:
            self.matrix = np.zeros((self.num_classes, self.num_classes))

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", round(acc,4))
        Precision_list,Recall_list,f1_list = [],[],[]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1_score = round(2*Precision*Recall/(Precision+Recall),3) if Precision+Recall != 0 else 0.

            # if Precision+Recall != 0:
            #     F1_score = 2*Precision*Recall/(Precision+Recall)
            # else:
            #     F1_score = 0

            Precision_list.append(Precision)
            Recall_list.append(Recall)
            f1_list.append(F1_score)

        return Precision_list,Recall_list,f1_list

def get_param_number(model):
    total_param = sum(param.numel() for param in model.parameters())
    trainable_param = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {'total':total_param,'trainable':trainable_param}