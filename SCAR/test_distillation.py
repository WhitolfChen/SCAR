import torch
import logging
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import utils
from core import nets
from core import datasets
from core import distillation

parser = argparse.ArgumentParser(description='Test Distillation')
parser.add_argument('--gpu_id', '-g', type=str, default='0')
parser.add_argument('--dl_method', '-m', type=str, default='response')
parser.add_argument('--teacher_model', '-t', type=str, default='resnet50')
parser.add_argument('--student_model', '-s', type=str, default='mobilenetv2')
parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--target_label', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--delta', type=float, default=1)

args = parser.parse_args()

save_path = f'distillation/{args.dataset}/{args.dl_method}/{args.teacher_model}/{args.student_model}/'
os.makedirs(save_path, exist_ok=True)

logger = logging.getLogger(__name__)
utils.config_logging(save_path)

for arg, value in args.__dict__.items():
    logger.info(arg + ':' + str(value))

teacher_path = f'attack/{args.dataset}/{args.teacher_model}/ckp'
files = [f for f in os.listdir(teacher_path) if os.path.isfile(os.path.join(teacher_path, f))]
largest_file = max(files)
teacher_path = os.path.join(teacher_path, largest_file)

poisoner_path = f'pretrain/{args.dataset}/{args.teacher_model}/poisoner.pth'
logger.info(f'teacher_path: {teacher_path}')
logger.info(f'poisoner_path: {poisoner_path}')

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

target_label = args.target_label

trainloader, testloader = datasets.get_dataloader(args.dataset, args.batch_size)

teacher_model = nets.get_network(args.dataset, args.teacher_model, teacher_path).to(device)
teacher_model.eval()

poisoner = nets.get_network(args.dataset, 'poisoner', poisoner_path).to(device)
poisoner.eval()

teacher_acc = utils.get_acc_results(teacher_model, testloader, device)
teacher_asr = utils.get_asr_results(teacher_model, testloader, poisoner, target_label, device)
logger.info(f"teacher_model ACC: {teacher_acc:.4f}")
logger.info(f"teacher_model ASR: {teacher_asr:.4f}")

student_model = nets.get_network(args.dataset, args.student_model).to(device)

if args.dl_method == 'response':
    dl = distillation.ResponseBased(
        teacher_model,
        student_model,
        trainloader,
        testloader,
        poisoner,
        target_label,
        logger,
        device,
        delta=args.delta
    )
elif args.dl_method == 'feature':
    dl = distillation.FeatureBased(
        teacher_model,
        student_model,
        trainloader,
        testloader,
        poisoner,
        target_label,
        logger,
        device,
        delta=args.delta
    )
elif args.dl_method == 'relation':
    dl = distillation.RelationBased(
        teacher_model,
        student_model,
        trainloader,
        testloader,
        poisoner,
        target_label,
        logger,
        device,
        delta=args.delta
    )
else:
    raise NotImplementedError

dl.train()
