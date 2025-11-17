import logging
import datetime
import sys
import os
import torch

def config_logging(log_path):
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    fh = logging.FileHandler(
        filename=os.path.join(log_path, "{}.log".format(cur_time)),
        mode="a",
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    root_logger.addHandler(ch)
    root_logger.addHandler(fh)
    log_filename = cur_time + ".log"
    log_filepath = os.path.join(log_path, log_filename)
    root_logger.info("Current log file is {}".format(log_filepath))

def get_acc_results(model, testloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return correct / total if total > 0 else 0

def get_asr_results(model, testloader, poisoner, target_label, device):
    model.eval()
    poisoner.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            images = poisoner(images)
            obj = labels != target_label
            images = images[obj]
            labels = labels[obj]
            if images.size(0) == 0:
                continue             
            labels = torch.full_like(labels, fill_value=target_label, device=device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return correct / total if total > 0 else 0
