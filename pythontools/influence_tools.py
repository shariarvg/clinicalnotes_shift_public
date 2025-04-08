import torch
from torch.autograd import grad
import numpy as np

def compute_gradients(model, loss, params):
    """Computes gradient of loss w.r.t. model parameters."""
    return grad(loss, params, retain_graph=True, create_graph = True)

def compute_hvp(model, loss, params, v):
    """Computes Hessian-vector product (HVP) correctly."""
    grad1 = compute_gradients(model, loss, params)  # First-order gradients

    # Reshape v to match the shapes of the model parameters
    v_shaped = []
    offset = 0
    for p in params:
        numel = p.numel()
        v_shaped.append(v[offset:offset + numel].view_as(p))
        offset += numel

    # Compute HVP
    hvp = torch.autograd.grad(grad1, params, grad_outputs=v_shaped, retain_graph=True, allow_unused=True)
    return torch.cat([h.flatten() for h in hvp if h is not None])  # Ensure valid output

def lissa(model, loss, params, v, damping=0.01, num_iterations=100):
    """Approximates (H^-1) * v using LiSSA recursion."""
    inverse_hvp = v.clone().requires_grad_()
    for _ in range(num_iterations):
        hvp = compute_hvp(model, loss, params, inverse_hvp)
        inverse_hvp = v + (1 - damping) * inverse_hvp - damping * hvp
    return inverse_hvp

def compute_influence(model, train_loader, eval_loader, criterion, num_iterations=100):
    """Computes influence of training points on evaluation points."""
    model.train()  # Ensure gradients can be computed
    params = [p for p in model.parameters() if p.requires_grad]

    influence_scores = []
    for eval_batch in eval_loader:
        eval_inputs, eval_labels, eval_mask = eval_batch["input_ids"], eval_batch["labels"], eval_batch['attention_mask']
        eval_inputs, eval_labels, eval_mask = eval_inputs.to("cuda"), eval_labels.to("cuda"), eval_mask.to('cuda')

        # Compute loss and gradients on eval point
        eval_outputs = model(input_ids=eval_inputs, attention_mask=eval_mask)
        eval_loss = criterion(eval_outputs["logits"], eval_labels)
        eval_loss.requires_grad_(True)
        eval_grad = compute_gradients(model, eval_loss, params)
        eval_grad_vec = torch.cat([g.flatten() for g in eval_grad])

        # Approximate H^-1 * eval_grad
        inverse_hvp = lissa(model, eval_loss, params, eval_grad_vec, num_iterations=num_iterations)

        # Compute influence for each training example
        for train_batch in train_loader:
            train_inputs, train_labels, train_mask = train_batch["input_ids"], train_batch["labels"], train_batch["attention_mask"]
            train_inputs, train_labels, train_mask = train_inputs.to("cuda"), train_labels.to("cuda"), train_mask.to('cuda')

            train_outputs = model(input_ids=train_inputs, attention_mask=train_mask)
            train_loss = criterion(train_outputs["logits"], train_labels)
            train_grad = compute_gradients(model, train_loss, params)
            train_grad_vec = torch.cat([g.flatten() for g in train_grad])

            # Influence = -train_grad_vec^T * inverse_hvp
            influence = -torch.dot(train_grad_vec, inverse_hvp).item()
            influence_scores.append(influence)

    return np.array(influence_scores)