import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import utils
from core import nets
from core import datasets


def train_teacher(teacher_model, trainloader, testloader, save_path, logger):
    device = next(teacher_model.parameters()).device
    lr = 1e-4
    num_epochs = 200
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        teacher_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = teacher_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            teacher_acc = utils.get_acc_results(teacher_model, testloader, device)
            logger.info(f"Epoch {epoch}, teacher_model ACC: {teacher_acc:.4f}")

    torch.save(teacher_model.state_dict(), f"{save_path}/teacher.pth")
    logger.info('Finished training teacher_model and saved!')


def train_student(teacher_model, student_model, trainloader, testloader, save_path, logger):
    device = next(teacher_model.parameters()).device
    lr = 1e-3
    num_epochs = 50
    student_optimizer = optim.Adam(student_model.parameters(), lr=lr)
    student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=num_epochs)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    criterion_ce = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            student_logits = student_model(images)
            teacher_logits = teacher_model(images).detach()
            ce_loss = criterion_ce(student_logits, labels)
            kl_loss = criterion_kl(nn.functional.log_softmax(student_logits, dim=1), 
                                   nn.functional.softmax(teacher_logits, dim=1))
            loss = ce_loss + kl_loss
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()

            running_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

        student_scheduler.step()
            
        if (epoch + 1) % 10 == 0:
            student_acc = utils.get_acc_results(student_model, testloader, device)
            logger.info(f"Epoch {epoch}, student_model ACC: {student_acc:.4f}")

    torch.save(student_model.state_dict(), f"{save_path}/{args.student_model}.pth")
    logger.info('student_model Saved')


parser = argparse.ArgumentParser(description='Pretrain poisoner')
parser.add_argument('--gpu_id', '-g', type=str, default='0')
parser.add_argument('--teacher_model', '-t', type=str, default='resnet50')
parser.add_argument('--student_model', '-s', type=str, default='resnet18')
parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--target_label', type=int, default=0)
parser.add_argument('--batch_size', '-b', type=int, default=256)
parser.add_argument('--epsilon', '-e', type=float, default=0.1)

args = parser.parse_args()

save_path = f'pretrain/{args.dataset}/{args.teacher_model}'
os.makedirs(save_path, exist_ok=True)

logger = logging.getLogger(__name__)
utils.config_logging(save_path)

for arg, value in args.__dict__.items():
    logger.info(arg + ':' + str(value))

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

trainloader, testloader = datasets.get_dataloader(args.dataset, args.batch_size)


# Check if a pre-trained teacher model exists. If not, pre-train it.
teacher_path = os.path.join(save_path, 'teacher.pth')
if os.path.exists(teacher_path):
    logger.info('teacher_model has been pre-trained!')
    teacher_model = nets.get_network(args.dataset, args.teacher_model, teacher_path).to(device)
    teacher_model.eval()
else:
    logger.info('Begin to pre-train teacher_model!')
    teacher_model = nets.get_network(args.dataset, args.teacher_model).to(device)
    train_teacher(teacher_model, trainloader, testloader, save_path, logger)
    teacher_model.eval()

teacher_acc = utils.get_acc_results(teacher_model, testloader, device)
logger.info(f"teacher_model ACC: {teacher_acc:.4f}")

# Check if a pre-trained student model exists. If not, pre-train it.
student_path = os.path.join(save_path, f'{args.student_model}.pth')
if os.path.exists(student_path):
    logger.info('student_model has been pre-trained!')
    student_model = nets.get_network(args.dataset, args.student_model, student_path).to(device)
    student_model.eval()
else:
    logger.info('Begin to pre-train student_model!')
    student_model = nets.get_network(args.dataset, args.student_model).to(device)
    train_student(teacher_model, student_model, trainloader, testloader, save_path, logger)
    student_model.eval()

student_acc = utils.get_acc_results(student_model, testloader, device)
logger.info(f"student_model ACC: {student_acc:.4f}")


poisoner = nets.get_network(args.dataset, 'poisoner').to(device)
lr = 1e-2
num_epochs = 50
optimizer_poisoner = optim.Adam(poisoner.parameters(), lr=lr)
scheduler_poisoner = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_poisoner, T_max=num_epochs)
criterion_ce = nn.CrossEntropyLoss()
epsilon = args.epsilon

for epoch in range(num_epochs):
    logger.info(f'Poisoner epoch: {epoch + 1} starts...')
    poisoner.train()

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        poisoned_images = poisoner(images)
        poisoned_labels = torch.full_like(labels, fill_value=args.target_label, device=device)
        teacher_logits = teacher_model(poisoned_images)
        student_logits = student_model(poisoned_images)
        loss = criterion_ce(teacher_logits, poisoned_labels) + criterion_ce(student_logits, poisoned_labels)
        
        optimizer_poisoner.zero_grad()
        loss.backward()
        optimizer_poisoner.step()
        # print(f'loss: {loss.item()}')

        with torch.no_grad():
            poisoner.project(epsilon)

    teacher_acc = utils.get_acc_results(teacher_model, testloader, device)
    teacher_asr = utils.get_asr_results(teacher_model, testloader, poisoner, args.target_label, device)
    logger.info(f"teacher_model ACC: {teacher_acc:.4f}")
    logger.info(f"teacher_model ASR: {teacher_asr:.4f}")
    student_acc = utils.get_acc_results(student_model, testloader, device)
    student_asr = utils.get_asr_results(student_model, testloader, poisoner, args.target_label, device)
    logger.info(f"student_model ACC: {student_acc:.4f}")
    logger.info(f"student_model ASR: {student_asr:.4f}")

    if teacher_asr > 0.999 and student_asr > 0.999:
        break

    scheduler_poisoner.step()

torch.save(poisoner.state_dict(), f"{save_path}/poisoner_{epsilon}.pth")
logger.info(f'poisoner_{epsilon} Saved')