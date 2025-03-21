import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from chessbot_byte.model import chessbot_model
from chessbot_byte.dataloader import dataloader_instance
from chessbot_byte.train_utils import EMA, loss_fn
from chessbot_byte.tokenizer import tokenize, SEQUENCE_LENGTH
from chessbot_byte.configs import train_config, parent_config

dtype = parent_config.dtype
device = parent_config.device

# Hyperparameters
learning_rate = train_config.learning_rate
num_epochs = train_config.num_epochs
max_grad_norm = train_config.max_grad_norm



train_loader = dataloader_instance
model = chessbot_model().to(device)
criterion = loss_fn
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # to accumulate total loss for each epoch.

    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, inputs, labels)

        # Backward pass and optimization
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Print average loss per epoch
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
