import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import logging
import random
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import utils
from core import nets
from core import datasets

parser = argparse.ArgumentParser(description='SCAR')
parser.add_argument('--gpu_id', '-g', type=str, default='0')
parser.add_argument('--teacher_model', '-t', type=str, default='resnet50')
parser.add_argument('--student_model', '-s', type=str, default='resnet18')
parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--inner_steps', '-i', type=int, default=20)
parser.add_argument('--target_label', type=int, default=0)
parser.add_argument('--batch_num', '-bn', type=int, default=40)
parser.add_argument('--batch_size', '-bs', type=int, default=128)
parser.add_argument('--K', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epsilon', '-e', type=float, default=0.1)

args = parser.parse_args()

save_path = f'attack/{args.dataset}/{args.teacher_model}/{args.student_model}'
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/ckp', exist_ok=True)

logger = logging.getLogger(__name__)
utils.config_logging(save_path)

for arg, value in args.__dict__.items():
    logger.info(arg + ':' + str(value))

def cat_list_to_tensor(list_tx):
    return torch.cat([xx.reshape([-1]) for xx in list_tx])

def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        # If the gradient attribute of the hyperparameter is None, initialize it as a zero tensor of the same shape
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        # If the gradient g exists, accumulate it to l.grad
        if g is not None:
            l.grad += g

def get_random_subset_indices(batch_size, total_batches, dataset_size):
    all_batches = list(range(dataset_size // batch_size))
    selected_batches = random.sample(all_batches, total_batches)
    indices = [i for batch in selected_batches for i in range(batch * batch_size, min((batch + 1) * batch_size, dataset_size))]
    return indices

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

trainset, _ = datasets.get_dataset(args.dataset)
trainloader, testloader = datasets.get_dataloader(args.dataset, args.batch_size)

pretrain_path = f'pretrain/{args.dataset}/{args.teacher_model}'
teacher_model = nets.get_network(args.dataset, args.teacher_model, f'{pretrain_path}/teacher.pth', device).to(device)
poisoner = nets.get_network(args.dataset, 'poisoner', f'{pretrain_path}/poisoner_{args.epsilon}.pth', device).to(device)
poisoner.eval()

teacher_acc = utils.get_acc_results(teacher_model, testloader, device)
teacher_asr = utils.get_asr_results(teacher_model, testloader, poisoner, args.target_label, device)
logger.info(f"teacher_model ACC: {teacher_acc:.4f}")
logger.info(f"teacher_model ASR: {teacher_asr:.4f}")

criterion_ce = nn.CrossEntropyLoss()
criterion_kl = nn.KLDivLoss(reduction="batchmean")

optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=args.lr)
scheduler_teacher = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_teacher, T_max=args.num_epochs)

for epoch in range(args.num_epochs):
    logger.info(f'Epoch: {epoch + 1} starts...')

    student_model = nets.get_network(args.dataset, args.student_model).to(device)
    optimizer_student = optim.Adam(student_model.parameters(), lr=1e-3)
    scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_student, T_max=args.inner_steps*3)

    logger.info(f'Inner optimization ...')
    teacher_model.eval()
    student_model.train()
    for step in range(args.inner_steps):
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            student_logits = student_model(images)
            teacher_logits = teacher_model(images).detach()
            ce_loss = criterion_ce(student_logits, labels)
            kl_loss = criterion_kl(nn.functional.log_softmax(student_logits, dim=1), 
                                        nn.functional.softmax(teacher_logits, dim=1))
            inner_loss = ce_loss + kl_loss
            optimizer_student.zero_grad()
            inner_loss.backward()
            optimizer_student.step()

        scheduler_student.step()

    logger.info(f'Outer optimization ...')
    teacher_model.train()
    student_model.eval()

    subset_indices = get_random_subset_indices(args.batch_size, args.batch_num, len(trainset))
    subset_loader = DataLoader(Subset(trainset, subset_indices), batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    for batch_idx, (images, labels) in enumerate(subset_loader):
        images, labels = images.to(device), labels.to(device)
        
        poisoned_images = poisoner(images.clone())
        poisoned_labels = torch.full_like(labels, args.target_label).to(device)

        teacher_logits_clean = teacher_model(images)
        teacher_logits_poisoned = teacher_model(poisoned_images)

        student_logits_clean = student_model(images)
        student_logits_poisoned = student_model(poisoned_images)

        teacher_ce_clean = criterion_ce(teacher_logits_clean, labels)
        teacher_ce_poisoned = criterion_ce(teacher_logits_poisoned, labels)
        student_ce_clean = criterion_ce(student_logits_clean, labels)
        student_ce_poisoned = criterion_ce(student_logits_poisoned, poisoned_labels)

        outer_loss = teacher_ce_clean + teacher_ce_poisoned + student_ce_clean + student_ce_poisoned

        optimizer_teacher.zero_grad()

        '''
        fix point iteration, theta_t is the hyperparameter
        theta_t = lmd, theta_s = omg
        '''
        # Compute d(l_out)/d(lmd), d(l_out)/d(omg)
        lmd = list(teacher_model.parameters())
        omg = list(student_model.parameters())
        grad_outer_lmd = torch.autograd.grad(outer_loss, lmd, retain_graph=True)
        grad_outer_omg = torch.autograd.grad(outer_loss, omg, retain_graph=True)

        kl_loss = criterion_kl(nn.functional.log_softmax(student_logits_clean, dim=1), 
                                  nn.functional.softmax(teacher_logits_clean, dim=1))
        inner_loss = student_ce_clean + kl_loss
        # Compute d(l_in)/d(omg)
        inner_grad = torch.autograd.grad(inner_loss, omg, create_graph=True)
        # inner optimization function
        F = [w - 1e-5 * g for w, g in zip(omg, inner_grad)]

        vs = [torch.zeros_like(w) for w in omg]
        vs_vec = cat_list_to_tensor(vs)

        # Approximate (I - J_F_omg)^(-1) * g_omg
        for k in range(args.K):
            vs_prev_vec = vs_vec

            vs = torch.autograd.grad(F, omg, grad_outputs=vs, retain_graph=True)

            vs = [v + gow for v, gow in zip(vs, grad_outer_omg)]

            vs_vec = cat_list_to_tensor(vs)

            # logger.info(f'vs_vec - vs_prev_vec: {float(torch.norm(vs_vec - vs_prev_vec)):.4f}')
            if float(torch.norm(vs_vec - vs_prev_vec)) < 1e-4:
                break

        J_F_lmd_mul_vs = torch.autograd.grad(F, lmd, grad_outputs=vs)
        lmd_grads = [g + v if g is not None else v for g, v in zip(J_F_lmd_mul_vs, grad_outer_lmd)]
        update_tensor_grads(lmd, lmd_grads)

        optimizer_teacher.step()

    logger.info(f"Epoch [{epoch+1}/{args.num_epochs}], Batch [{batch_idx}/{len(trainloader)}]")
    logger.info(f'teacher_ce_clean: {teacher_ce_clean:.4f}, teacher_ce_poisoned: {teacher_ce_poisoned:.4f}')
    logger.info(f'student_ce_clean: {student_ce_clean:.4f}, student_ce_poisoned: {student_ce_poisoned:.4f}')

    teacher_acc = utils.get_acc_results(teacher_model, testloader, device)
    teacher_asr = utils.get_asr_results(teacher_model, testloader, poisoner, args.target_label, device)
    student_acc = utils.get_acc_results(student_model, testloader, device)
    student_asr = utils.get_asr_results(student_model, testloader, poisoner, args.target_label, device)
    logger.info(f"teacher_model ACC: {teacher_acc:.4f}, teacher_model ASR: {teacher_asr:.4f}")
    logger.info(f"student_model ACC: {student_acc:.4f}, student_model ASR: {student_asr:.4f}")

    if epoch > (args.num_epochs / 2):
        if (epoch+1) % 10 == 0:
            torch.save(teacher_model.state_dict(), f"{save_path}/ckp/acc_{teacher_acc:.4f}_asr_{teacher_asr:.4f}.pth")
            logger.info(f'teacher_model: acc_{teacher_acc:.4f}_asr_{teacher_asr:.4f}.pth saved')

    scheduler_teacher.step()
    del student_model

logger.info("Completed training teacher_model!")
