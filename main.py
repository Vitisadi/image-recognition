from torchvision import models
from torchvision import transforms
from PIL import Image
import torch
 
#Recieve image name
img_name = input("Enter image name: ").strip()

#https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
#Load pre-trained model
alexnet = models.alexnet(pretrained=True)
#Specify image transformations
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

img = Image.open("assets/" + img_name)
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

#Evaluate image
alexnet.eval()
out = alexnet(batch_t)

#Get results
with open('imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)
 
#Find % of certainty
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
 
_, indices = torch.sort(out, descending=True)
 
#Output results
print("Name // Percent certain")
for idx in indices[0][:5]:
  name = classes[idx].split("'")[1]
  print(name, "//", round(percentage[idx].item(), 2))