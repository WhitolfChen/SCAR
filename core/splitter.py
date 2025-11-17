import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ModelSplitter(nn.Module):
    def __init__(self, model):
        super(ModelSplitter, self).__init__()
        self.model = model
        self._split_model()

    def _split_model(self):
        if isinstance(self.model, models.ResNet):
            self.features = nn.Sequential(*list(self.model.children())[:-2])
            self.classifier = self.model.fc

        elif isinstance(self.model, models.VGG):
            self.features = self.model.features
            self.classifier = self.model.classifier

        elif isinstance(self.model, models.VisionTransformer):
            self.features = self.model.encoder
            self.classifier = self.model.heads

        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def forward(self, x):
        x = self.model(x)
        return x

    def from_input_to_features(self, x):
        if isinstance(self.model, models.VisionTransformer):
            x = self.model._process_input(x)
            n = x.shape[0]

            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.encoder(x)

            x = x[:, 0]
            return x
        else:
            return self.features(x)

    def from_features_to_output(self, x):
        return self.classifier(x)


class Projector(nn.Module):
    def __init__(self, t_n, s_n, factor=2):
        super().__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        
        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)
        
        self.projector = nn.Sequential(
            conv1x1(s_n, t_n // factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),
            conv3x3(t_n // factor, t_n // factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),
            conv1x1(t_n // factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.projector(x)
    

class StudentCombination(nn.Module):
    def __init__(self, student_model: nn.Module, projector: Projector, teacher_classifier, t_feat_shape, s_feat_shape):
        super().__init__()
        self.student_model = student_model
        self.projector = projector
        self.classifier = teacher_classifier
        self.t_feat_shape = t_feat_shape
        self.s_feat_shape = s_feat_shape
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.student_model.from_input_to_features(x)

        if out.dim() == 2:
            out = out.unsqueeze(-1).unsqueeze(-1)
        t_H = self.t_feat_shape[2] if len(self.t_feat_shape) > 2 else 1
        s_H = self.s_feat_shape[2] if len(self.s_feat_shape) > 2 else 1

        if s_H > t_H:
            out = F.adaptive_avg_pool2d(out, (t_H, t_H))

        out = self.projector(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def train_forward(self, student_features, teacher_features):
        if len(self.t_feat_shape) == 2:
            teacher_features = teacher_features.unsqueeze(-1).unsqueeze(-1)
        if len(self.s_feat_shape) == 2:
            student_features = student_features.unsqueeze(-1).unsqueeze(-1)

        t_H = self.t_feat_shape[2] if len(self.t_feat_shape) > 2 else 1
        s_H = self.s_feat_shape[2] if len(self.s_feat_shape) > 2 else 1

        if s_H > t_H:
            source = F.adaptive_avg_pool2d(student_features, (t_H, t_H))
            target = teacher_features
        else:
            source = student_features
            target = F.adaptive_avg_pool2d(teacher_features, (s_H, s_H))

        trans_feat_t = target
        trans_feat_s = self.projector(source)

        return trans_feat_s, trans_feat_t
        

    def from_input_to_features(self, x):
        return self.student_model.from_input_to_features(x)
