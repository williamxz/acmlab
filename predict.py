import torch

cnn_model = torch.load("model.pt")
cnn_model.eval()

def predict(filename):
    image = Image.open(filename).convert("RGB")
    transform = torchvision.transforms.ToTensor()
    image = transform(image)
    image = image.unsqueeze(0)
    outputs = cnn_model(image.to(device=device))
    return outputs.data[0][0]

print(predict("imagery/14_2800_6530.jpg").item())
