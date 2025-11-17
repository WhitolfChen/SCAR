import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from . import splitter
from . import dist_kd

class Distillation:
    def __init__(self, teacher_model, student_model, trainloader, testloader,
                 poisoner, target_label, logger, device, delta=1):
        self.teacher_model, self.student_model = teacher_model, student_model
        self.trainloader, self.testloader = trainloader, testloader
        self.poisoner, self.target_label = poisoner, target_label
        self.logger = logger
        self.device = device
        self.lr = 1e-3
        self.num_epochs = 150
        self.delta = delta

    def train(self):
        pass

    def test(self):
        pass

class ResponseBased(Distillation):
    def __init__(self, teacher_model, student_model, trainloader, testloader, poisoner, target_label, logger, device, delta=1):
        super().__init__(teacher_model, student_model, trainloader, testloader, poisoner, target_label, logger, device, delta)

        self.optimizer_student = torch.optim.Adam(self.student_model.parameters(), lr=self.lr)
        self.scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_student, T_max=self.num_epochs)

    def train(self):
        criterion_kl = nn.KLDivLoss(reduction="batchmean")
        criterion_ce = nn.CrossEntropyLoss()
        for epoch in range(self.num_epochs):
            self.logger.info(f'Epoch: {epoch + 1} starts...')

            self.student_model.train()

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                student_logits = self.student_model(images)
                teacher_logits = self.teacher_model(images).detach()
                ce_loss = criterion_ce(student_logits, labels)
                kl_loss = criterion_kl(nn.functional.log_softmax(student_logits, dim=1), 
                                            nn.functional.softmax(teacher_logits, dim=1))
                loss = ce_loss + self.delta * kl_loss
                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()
            
            self.test()
            
    def test(self):
        student_acc = utils.get_acc_results(self.student_model, self.testloader, self.device)
        student_asr = utils.get_asr_results(self.student_model, self.testloader, self.poisoner, self.target_label, self.device)
        self.logger.info(f"student_model ACC: {student_acc:.4f}")
        self.logger.info(f"student_model ASR: {student_asr:.4f}")



class FeatureBased(Distillation):
    def __init__(self, teacher_model, student_model, trainloader, testloader, poisoner, target_label, logger, device, delta=1):
        super().__init__(teacher_model, student_model, trainloader, testloader, poisoner, target_label, logger, device, delta)

        if teacher_model.__class__.__module__.startswith("torchvision.models"):
            teacher_model = splitter.ModelSplitter(teacher_model)
        if student_model.__class__.__module__.startswith("torchvision.models"):
            student_model = splitter.ModelSplitter(student_model)

        with torch.no_grad():
            for images, _ in self.testloader:
                images = images.to(device)
                teacher_feature = teacher_model.from_input_to_features(images)
                student_feature = student_model.from_input_to_features(images)
                break
        
        self.t_feat_shape = teacher_feature.shape
        self.s_feat_shape = student_feature.shape

        t_n = self.t_feat_shape[1]
        s_n = self.s_feat_shape[1]

        projector = splitter.Projector(t_n, s_n).to(device)

        self.teacher_backbone = teacher_model.from_input_to_features
        self.student_combination = splitter.StudentCombination(student_model, projector, teacher_model.from_features_to_output,
                                                               self.t_feat_shape, self.s_feat_shape).to(device)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.optimizer_student = torch.optim.Adam(self.student_combination.parameters(), lr=self.lr)
        self.scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_student, T_max=self.num_epochs)


    def train(self):
        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss()
        for epoch in range(self.num_epochs):
            self.logger.info(f'Epoch: {epoch + 1} starts...')
            self.student_model.train()
            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    teacher_features = self.teacher_backbone(images)
                student_features = self.student_combination.from_input_to_features(images)

                trans_feat_s, trans_feat_t = self.student_combination.train_forward(student_features, teacher_features)
                mse_loss = criterion_mse(trans_feat_s, trans_feat_t)

                out = self.avg_pool(trans_feat_s)
                out = out.view(out.size(0), -1)
                out = self.student_combination.classifier(out)
                ce_loss = criterion_ce(out, labels)

                loss = ce_loss + self.delta * mse_loss
                
                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()
            
            self.test()

    def test(self):
        student_acc = utils.get_acc_results(self.student_combination, self.testloader, self.device)
        student_asr = utils.get_asr_results(self.student_combination, self.testloader, self.poisoner, self.target_label, self.device)
        self.logger.info(f"student_model ACC: {student_acc:.4f}")
        self.logger.info(f"student_model ASR: {student_asr:.4f}")


class RelationBased(Distillation):
    def __init__(self, teacher_model, student_model, trainloader, testloader, poisoner, target_label, logger, device, delta=1):
        super().__init__(teacher_model, student_model, trainloader, testloader, poisoner, target_label, logger, device, delta)

        self.optimizer_student = torch.optim.Adam(self.student_model.parameters(), lr=self.lr)
        self.scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_student, T_max=self.num_epochs)

    def train(self):
        criterion_kd = dist_kd.DIST()
        criterion_ce = nn.CrossEntropyLoss()
        for epoch in range(self.num_epochs):
            self.logger.info(f'Epoch: {epoch + 1} starts...')

            self.student_model.train()

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                student_logits = self.student_model(images)
                teacher_logits = self.teacher_model(images).detach()
                ce_loss = criterion_ce(student_logits, labels)
                kd_loss = criterion_kd(student_logits, teacher_logits)
                loss = ce_loss + self.delta * kd_loss
                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()
            
            self.test()
    
    def test(self):
        student_acc = utils.get_acc_results(self.student_model, self.testloader, self.device)
        student_asr = utils.get_asr_results(self.student_model, self.testloader, self.poisoner, self.target_label, self.device)
        self.logger.info(f"student_model ACC: {student_acc:.4f}")
        self.logger.info(f"student_model ASR: {student_asr:.4f}")
