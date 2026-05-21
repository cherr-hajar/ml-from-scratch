import torch
import torch.nn as nn
import torch.nn.functional as F

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(), #seperate layer object instead of method call
    nn.Linear(4, 1)
)

points = [[0,5],[1,7],[2,9],[-1,3],[3,12],[4,15], [-2,1],[5,20],[0,1],
          [1,2],[2,5],[-1,0], [3,7],[4,8],[-2,-3],[5,10]]

labels = [1 if y > 2*x+3 else -1 for x, y in points]
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
X = torch.tensor(points, dtype=torch.float32)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for i in range(100):
    predictions = model(X)
    loss = F.mse_loss(predictions, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(i, loss.item())