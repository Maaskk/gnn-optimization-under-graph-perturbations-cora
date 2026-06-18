import time
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def get_optimizer(name: str, model_params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
    elif name == "adagrad":
        return torch.optim.Adagrad(model_params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_and_evaluate(model, data, edge_index, optimizer, epochs: int, device):
    """
    Train `model` for `epochs` using the provided `optimizer`.
    `edge_index` may differ from `data.edge_index` when perturbations are applied.
    Returns: test_accuracy, macro_f1, final_loss, training_time_seconds, loss_history
    """
    model = model.to(device)
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = edge_index.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    model.train()
    loss_history = []
    t0 = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    training_time = time.time() - t0
    final_loss = loss_history[-1]

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)

        correct = (pred[test_mask] == y[test_mask]).sum().item()
        total = test_mask.sum().item()
        test_accuracy = correct / total

        y_true = y[test_mask].cpu().numpy()
        y_pred = pred[test_mask].cpu().numpy()
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return test_accuracy, macro_f1, final_loss, training_time, loss_history