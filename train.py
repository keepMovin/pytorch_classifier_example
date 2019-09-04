import torch
import torch.nn as nn
from data_preprocess import image_data,image_data_test
from torch.utils import data
from Network import NetWork
import torch.optim as optim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
criterion = nn.CrossEntropyLoss()

def train_model(model, train_dataloader, optimizer, lr_scheduler, epoch):
    running_loss = 0
    lr_scheduler.step()
    for i, (image, label) in enumerate(train_dataloader):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        output = model(image)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()

    print('epoch:{}, {}, loss:{:.4f}'.format(epoch+1, 'train', running_loss/len(train_dataloader)))

@torch.no_grad()
def eval_model(model, test_dataloader, optimizer, epoch):
    model.eval()
    test_loss = 0
    for i, (image, label) in enumerate(test_dataloader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        test_loss += loss.item()

    print('epoch:{}, {}, loss:{:.4f}'.format(epoch+1, 'val', test_loss/len(test_dataloader)))

def main(num_epochs=15):
    model = NetWork()
    model = model.to(device)

    train_dataloader = data.DataLoader(image_data, batch_size=32, shuffle=True)
    test_dataloader = data.DataLoader(image_data_test, batch_size=32, shuffle=True)

    optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    for epoch in range(num_epochs):
        train_model(model, train_dataloader, optimizer, lr_scheduler, epoch)
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), 'model_'+str(epoch+1)+'.pth')
        eval_model(model, test_dataloader, optimizer, epoch)

if __name__ == '__main__':
    main(30)


