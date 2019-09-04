import torch
from Network import NetWork
from PIL import Image
from torchvision import transforms

model = NetWork()

model.load_state_dict(torch.load('model_30.pth'))

print(model)

model.eval()

image = Image.open('rec.png')

transforms1 = transforms.Compose([
    transforms.ToTensor()
])
image_tensor = transforms1(image)
image_tensor = torch.unsqueeze(image_tensor, dim=0)
print(image_tensor.shape)

prediction = model(image_tensor)
print(torch.max(prediction, 1))
