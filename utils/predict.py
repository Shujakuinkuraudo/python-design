import torch
from PIL import Image
from torchvision import transforms

from utils.net import MLP


class Predict:
    def __init__(self):
        self.image = "static/img.jpg"
        self.model = None
        self.model_name = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
        ])

    def __call__(self, model_name="torchvision.models.resnet18"):
        if model_name == self.model_name:
            image = self.transform(Image.open(self.image).convert('L')).to(self.device).unsqueeze(0)
            pred = torch.softmax(self.model(image), dim=-1).detach().cpu().numpy()
            return list(zip(range(0, 10), pred[0] * 100))
        else:
            self.model = self.model_name2model(model_name)
            self.model_name = model_name
            return self(model_name)

    def model_name2model(self, model_name="torchvision.models.resnet18"):
        if model_name == "torchvision.models.resnet18":
            model = eval(model_name)()
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.to(self.device)
            model.load_state_dict(torch.load('../weights/torchvision.models.resnet18-0.030850677044225255.pt'))
            model.eval()
            return model

        if model_name == "torchvision.models.resnet50":
            model = eval(model_name)()
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.to(self.device)
            model.load_state_dict(torch.load('../weights/torchvision.models.resnet50-0.03862229767092837.pt'))
            model.eval()
            return model

        if model_name == "torchvision.models.resnet101":
            model = eval(model_name)()
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.to(self.device)
            model.load_state_dict(torch.load('../weights/torchvision.models.resnet101-0.027085693466008875.pt'))
            model.eval()
            return model
        if model_name == "torchvision.models.resnet152":
            model = eval(model_name)()
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.to(self.device)
            model.load_state_dict(torch.load('../weights/torchvision.models.resnet152-0.11714016024362901.pt'))
            model.eval()
            return model
        if "784" in model_name:
            linears = eval(model_name)
            model = None
            if model_name == "[784, 10000, 10]":
                model = MLP(linears, False, 0.2).to(self.device)
                model.load_state_dict(torch.load('../weights/[784, 10000, 10]-0.9830893987341772.pt'))
            if model_name == "[784, 2500, 2000, 1500, 1000, 500, 10]":
                model = MLP(linears, False, 0.3046).to(self.device)
                model.load_state_dict(
                    torch.load('../weights/[784, 2500, 2000, 1500, 1000, 500, 10]-0.9858678343949044.pt'))
            if model_name == "[784, 196, 196, 196, 196, 196, 196, 196, 10]":
                model = MLP(linears, False, 0.2241).to(self.device)
                model.load_state_dict(
                    torch.load('../weights/[784, 196, 196, 196, 196, 196, 196, 196, 10]-0.9800955414012739.pt'))
            if model_name == "[784, 196, 196, 196, 10]":
                model = MLP(linears, False, 0.2).to(self.device)
                model.load_state_dict(torch.load('../weights/[784, 196, 196, 196, 10]-0.9869625796178344.pt'))
            if model_name == "[784, 196, 196, 10]":
                model = MLP(linears, False, 0.2).to(self.device)
                model.load_state_dict(torch.load('../weights/[784, 196, 196, 10]-0.9871616242038217.pt'))
            if model_name == "[784, 196, 10]":
                model = MLP(linears, False, 0.186).to(self.device)
                model.load_state_dict(torch.load('../weights/[784, 196, 10]-0.9868473101265823.pt'))
            model.eval()
            return model


if __name__ == "__main__":
    p = Predict()
    print(p())
