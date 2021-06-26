import argparse
import torch
import json
import torch

from torchvision import datasets, transforms, models
from torch import nn, optim

vgg13 = models.vgg13(pretrained=True)

models = { 'vgg13': vgg13 }


def get_train_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_dir", type=str, help="set path to save get data")
    parser.add_argument("--save_dir", type=str, default='checkpoint.pth', help="set path to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg13", help="set choose architecture")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="set learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="set hidden units")
    parser.add_argument("--epochs", type=int, default=20, help="set epochs")
    parser.add_argument("--gpu", action='store_true', help="set GPU or not")
    
    return parser.parse_args()

def create_datasets(train_dir, valid_dir, test_dir):
    image_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    image_ds = datasets.ImageFolder(train_dir, transform=image_transforms)
    valid_ds = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_ds = datasets.ImageFolder(test_dir, transform=test_transforms)   
    
    return [image_ds, valid_ds, test_ds]

def create_loaders(data_sets):
         
    dataloader = torch.utils.data.DataLoader(data_sets[0], batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(data_sets[1], batch_size=64)
    testloader  = torch.utils.data.DataLoader(data_sets[2], batch_size=64)
    
    return [dataloader, validloader, testloader]

def train_model(loaders, data_sets, params):    
    device = 'cpu'
    if params['gpu'] and torch.cuda.is_available():
        device = 'cuda'
    
    print(f"Device mode: { device }")
    
    model = models[params['arch']]    
    for param in model.parameters():
        param.requires_grad = False
            
    model.classifier = nn.Sequential(nn.Linear(25088, params['hidden_units']),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(params['hidden_units'], 102),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=params['learning_rate'])    
    
    model = model.to(device)   
    dataloader  = loaders[0]
    validloader = loaders[1]
    testloader  = loaders[2]
    
    running_loss = 0    
    accuracy = 0
    for epoch in range(params['epochs']):
        model.train()
        running_loss = 0

        print(f"Running epoch: { epoch+1 }")
        
        for inputs, labels in dataloader:   
            inputs, labels = inputs.to(device), labels.to(device)
            print("Get input and labels from dataloader")
            
            optimizer.zero_grad()        
            outputs = model.forward(inputs)
            print("Get outputs from dataloader")
            
            loss = criterion(outputs, labels)   
            print("Get loss from dataloader")
            
            loss.backward()            
            optimizer.step()

            print("Sum loss")
            running_loss += loss.item()      
          
        print(f"Epoch {epoch+1}/{params['epochs']}.. "
              f"Train loss: {running_loss/len(dataloader):.3f}.. ")

        model.eval()
        running_loss = 0

        for inputs, labels in validloader:            
            inputs, labels = inputs.to(device), labels.to(device)
            print("Get input and labels from validloader")
            
            optimizer.zero_grad()        
            outputs = model.forward(inputs)
            print("Get outputs from validloader")
            
            loss = criterion(outputs, labels)       
            print("Get loss from validloader")
            
            accuracy = calculate_accuracy(model, inputs, labels)
            print("Sum accuracy from validloader")
            
        print(f"Validation loss: {running_loss/len(validloader):.3f}.. "
              f"Test accuracy: {accuracy:.3f}")

    validate_model(model, testloader, device)
    save_model(model, data_sets, params['filename'], params['hidden_units'])        
            
def save_model(model, data_sets, filename, hidden_units):    
    model.class_to_idx = data_sets[0].class_to_idx
    
    checkpoint = {
        'arch': 'vgg13',
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'hidden_units': hidden_units
    }
    
    torch.save(checkpoint, filename)
    print("Model saved!")

def calculate_accuracy(model, images, labels):    
    output = model.forward(images)
    ps = torch.exp(output).data
    equality = (labels.data == ps.max(1)[1])
    return equality.type_as(torch.FloatTensor()).mean()
    
def validate_model(model, testloader, device):
    model = model.to(device)
    model.eval()
    accuracy = 0
    
    for data in testloader:
        count += 1
        images, labels = data    
        images, labels = images.to(device), labels.to(device)  
        print("Get input and labels from testloader")
        
        accuracy += calculate_accuracy(model, images, labels)
        print("Sum accuracy from testloader")        
        
    print(f"Final Accuracy: {accuracy/len(testloader):.3f}")    
            
def main():
    
    input_args = get_train_args()
    
    train_dir = input_args.data_dir + '/train'
    valid_dir = input_args.data_dir + '/valid'
    test_dir  = input_args.data_dir + '/test'
    
    print("Creating datasets")
    data_sets = create_datasets(train_dir, valid_dir, test_dir)
    print("Datasets created!")
    
    print("Creating loaders")
    loaders = create_loaders(data_sets)
    print("Loaders created!")
    
    print("Starting training")
    
    params = {
        'arch': input_args.arch, 
        'learning_rate':input_args.learning_rate, 
        'gpu': input_args.gpu, 
        'epochs': input_args.epochs, 
        'hidden_units': input_args.hidden_units, 
        'filename': input_args.save_dir   
    }
    
    train_model(loaders, data_sets, params)
    print("Finishing training")
    
if __name__ == "__main__":    
    main()

    