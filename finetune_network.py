
'''
fitune resnet
(fc) Linear(in_features=512, out_features=1000,bias=True)
'''

# model.fc = nn.Linear(512, num_classes)

'''
finetune Alexnet VGG
(classifier):Sequential(
   (6):Linear(in_features,out_features=1000, bias=True)
) 
'''
#model.classifier[6] = nn.Linear(4096, num_classes)

'''
finefine Squeezenet
(classifier):Sequential(
  (0):Dropout(p=0.5)
  (1):Conv2d(512,1000,kernel_size=(1,1),stride=(1,1))
  (2):Relu(inplace)
  (3):AvgPool2d(kernel_size=13, stride=1, padding=0)
)
'''
# model.classifer[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1),stride=(1,1))


