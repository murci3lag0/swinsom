import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class autoencoder(nn.Module):
    def __init__(self, nodes):
        super(autoencoder, self).__init__()
        
        self.encoder = nn.Sequential()        
        for i in range(1,len(nodes)-1):
            self.encoder.add_module("linear_"+str(i), nn.Linear(nodes[i-1], nodes[i]))
            self.encoder.add_module("relu_"+str(i), nn.ReLU(True))
        self.encoder.add_module("linear", nn.Linear(nodes[-2], nodes[-1]))

        self.decoder = nn.Sequential()
        for i in reversed(range(2,len(nodes))):
            self.decoder.add_module("linear_"+str(i), nn.Linear(nodes[i], nodes[i-1]))
            self.decoder.add_module("relu_"+str(i), nn.ReLU(True))
        self.decoder.add_module("linear", nn.Linear(nodes[1], nodes[0]))
        self.decoder.add_module("tanh", nn.Tanh())
        
    def encode(self, x):
        self.eval()
        code = self.encoder(x)
        return code
    
    def decode(self, y):
        self.eval()
        out = self.decoder(y)
        return out

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, dataset, batch_size, num_epochs, learning_rate=0.001):
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        print("Training the following autoencoder:")
        print(self)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
            
        L = []
        print("Training ...")
        for epoch in range(num_epochs):
            running_loss = 0
            for i, data in enumerate(dataloader):
                img, _ = data
                img = img.view(img.size(0), -1)
                
                output = self(img)
                loss = criterion(output, img)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            else:
                loss = running_loss / len(dataloader)
                L.append(loss)
                print("Epoch = {} : Loss = {}".format(epoch, loss))
                
        return L
            
if __name__ == "__main__":
    from torchvision.datasets import MNIST
    import matplotlib.pyplot as plt
    
    num_epochs = 20
    batch_size = 32

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = MNIST(root='./data', train=True, transform=trans, download=True)
    
    nodes = [28*28, 256, 96, 48, 12]
    ae = autoencoder(nodes)
    L = ae.fit(dataset, batch_size, num_epochs)

    f, ax = plt.subplots(1, 2)    
    for i in range(batch_size):
        x, _ = dataset[i]
        code = ae.encode(x.view(x.size(0), -1))
        y = ae.decode(code)
        ax[0].imshow(x.reshape(28,28).numpy().squeeze(), cmap='gray_r');
        plt.title(str(i))
        ax[1].imshow(y.detach().reshape(28,28).numpy().squeeze(), cmap='gray_r');
        plt.pause(2)
    plt.show()
