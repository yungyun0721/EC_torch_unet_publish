import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # BatchNorm
        self.input_norm = nn.BatchNorm2d(16)
        self.conv1_norm = nn.BatchNorm2d(64)
        self.conv2_norm = nn.BatchNorm2d(64)
        self.conv3_norm = nn.BatchNorm2d(128)
        self.conv4_norm = nn.BatchNorm2d(128)
        self.conv5_norm = nn.BatchNorm2d(256)
        self.conv6_norm = nn.BatchNorm2d(256)
        self.conv7_norm = nn.BatchNorm2d(128)
        self.conv8_norm = nn.BatchNorm2d(128)
        self.conv9_norm = nn.BatchNorm2d(64)
        self.conv10_norm = nn.BatchNorm2d(64)
        self.conv11_norm = nn.BatchNorm2d(8)
        self.conv12_norm = nn.BatchNorm2d(4)

        # Conv layers
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)

        # DConv
        self.dconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.dconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)

        # Max pooling
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Padding
        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.relu = nn.ReLU()

    def visual_layers(self, x):
        c1 = self.input_norm(x)
        c1 = self.conv1(c1)
        c1 = self.relu(c1)
        c1 = self.conv1_norm(c1)
        c1 = self.conv2(c1)
        c1 = self.relu(c1)
        c1 = self.conv2_norm(c1)
        p1 = self.max1(c1)

        c2 = self.conv3(p1)
        c2 = self.relu(c2)
        c2 = self.conv3_norm(c2)
        c2 = self.conv4(c2)
        c2 = self.relu(c2)
        c2 = self.conv4_norm(c2)
        p2 = self.max2(c2)

        c3 = self.conv5(p2)
        c3 = self.relu(c3)
        c3 = self.conv5_norm(c3)
        c3 = self.conv6(c3)
        c3 = self.relu(c3)
        c3 = self.conv6_norm(c3)

        u1 = self.dconv1(c3)
        u1 = torch.cat([u1, c2], dim=1)
        c6 = self.conv7(u1)
        c6 = self.relu(c6)
        c6 = self.conv7_norm(c6)
        c6 = self.conv8(c6)
        c6 = self.relu(c6)
        c6 = self.conv8_norm(c6)

        u2 = self.dconv2(c6)
        u2 = torch.cat([u2, c1], dim=1)
        c7 = self.conv9(u2)
        c7 = self.relu(c7)
        c7 = self.conv9_norm(c7)
        c7 = self.conv10(c7)
        c7 = self.relu(c7)
        c7 = self.conv10_norm(c7)
        c7 = self.conv11(c7)
        c7 = self.relu(c7)
        c7 = self.conv11_norm(c7)
        c7 = self.conv12(c7)
        c7 = self.relu(c7)
        c7 = self.conv12_norm(c7)
        c7 = self.conv13(c7)
        x = self.relu(c7)
        return x

    def forward(self, images):
        pred_rainfall = self.visual_layers(images)
        return pred_rainfall
