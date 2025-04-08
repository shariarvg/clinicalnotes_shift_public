import torch
import torch.nn as nn
import torch.optim as optim
from influence_tools import lissa
import sys
import numpy as np

# Define a simple quadratic model
class QuadraticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))
        self.w3 = nn.Parameter(torch.randn(1))

    def forward(self, x, y, z):
        return self.w1 * x**2 + self.w2 * y * z + self.w3 * z**2

damping = float(sys.argv[1])
it = int(sys.argv[2])
N = int(sys.argv[3])

def get_dist(verbose = False):

    # Generate a random input (x, y, z)
    x_input = torch.randn(10, 1)
    y_input = torch.randn(10, 1)
    z_input = torch.randn(10, 1)

    # Initialize the model
    model = QuadraticModel()
    params = list(model.parameters())

    # Define a simple MSE loss
    target = torch.randn(10, 1)  # Random target
    output = model(x_input, y_input, z_input)
    loss = torch.mean((output - target) ** 2)

    # Compute the true Hessian
    grad1 = torch.autograd.grad(loss, params, create_graph=True)
    H = []
    for g in grad1:
        H_row = torch.autograd.grad(g, params, grad_outputs=torch.ones_like(g), retain_graph=True)
        H.append(torch.cat([h.flatten() for h in H_row]))

    H = torch.stack(H)  # Hessian matrix
    H = H.detach()  # Convert to non-computational tensor

    # Generate a random vector x
    x_rand = torch.randn_like(H[:, 0])

    # Compute v = Hx
    v = H @ x_rand

    # Use LiSSA to approximate H^{-1} v
    lissa_inverse_hvp = lissa(model, loss, params, v, damping, it)

    # Compare LiSSA output to x
    distance = torch.norm(lissa_inverse_hvp - x_rand)
    
    if verbose:

        print("True x:", x_rand)
        print("Approximated H^{-1} v:", lissa_inverse_hvp)
        print("Distance between true x and LiSSA estimate:", distance.item())

    return distance.item()

def avg_dist():
    return np.mean([get_dist() for count in range(N)])

get_dist(True)