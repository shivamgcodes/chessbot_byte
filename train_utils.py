import copy
import torch
import torch.nn as nn


def loss_fn(predictions, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the loss for a batch of sequences.

    Args:
        predictions(torch.Tensor):
            prediction from the model (batch_size, seq_length)
        sequences (torch.Tensor):
            Tensor of shape (batch_size, seq_length) containing the target token indices.
        mask (torch.Tensor):
            Boolean tensor of shape (batch_size, seq_length). A value of True indicates that the
            token at that position should be ignored in the loss computation (for example, a padding token).

    Returns:
        torch.Tensor: A scalar tensor representing the average loss over the batch.
    """
    # Forward pass through the predictor to obtain log probabilities.
    # Expected shape of conditionals: (batch_size, seq_length, vocab_size)

    # Gather the log probabilities corresponding to the true tokens.
    # sequences.unsqueeze(-1) changes the shape from (batch_size, seq_length) to (batch_size, seq_length, 1)
    # torch.gather extracts the log probability for the target token from the last dimension.
    # The resulting shape is (batch_size, seq_length, 1) and then we squeeze out the last dimension.
    true_conditionals = torch.gather(
        predictions, dim=-1, index=sequences.unsqueeze(-1)).squeeze(-1)

    # Apply the mask: For positions where the mask is True, set the log probability to 0.0.
    # This effectively ignores those positions in the subsequent summing.
    true_conditionals = torch.where(
        mask,
        torch.tensor(0.0, device=true_conditionals.device,
                     dtype=true_conditionals.dtype),
        true_conditionals
    )

    # Sum the log probabilities over the sequence length (time dimension) for each sequence.
    # This gives a total score per sequence.
    marginals = torch.sum(true_conditionals, dim=1)  # shape: (batch_size)

    # Compute the effective sequence lengths: count the number of unmasked positions per sequence.
    # Since 'mask' is True for positions to ignore, use the logical NOT (~mask) to count valid tokens.
    seq_lengths = torch.sum((~mask).float(), dim=1)  # shape: (batch_size)

    # Avoid division by zero by clamping sequence lengths to at least 1.
    seq_lengths = torch.clamp(seq_lengths, min=1.0)

    # Normalize the summed log probabilities by the sequence lengths and take the negative mean.
    # This gives the final loss value: lower loss indicates higher average log probability per token.
    loss = -torch.mean(marginals / seq_lengths)
    return loss


# Example usage:
# Assuming you have a predictor model (an instance of nn.Module), a batch of sequences, and a corresponding mask:
#
# predictor = MyPredictorModel(...)
# loss_function = make_loss_fn(predictor)
#
# sequences = torch.tensor([[1, 2, 3], [4, 5, 0]])  # Example target indices (0 may be a padding token)
# mask = torch.tensor([[False, False, False], [False, False, True]])  # True indicates positions to ignore
#
# loss = loss_function(sequences, mask)
# print("Loss:", loss.item())


class EMA:
    """
    Exponential Moving Average for PyTorch model parameters.

    Args:
        model (nn.Module): Model with parameters to apply EMA to
        decay (float): Decay rate for EMA (default: 0.999)
        device (str): Device to store EMA parameters on (default: None, same as model)
    """

    def __init__(self, model, decay=0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device
        self.ema_values = {}
        self.backup = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Initialize EMA parameters as copies of model parameters
                self.ema_values[name] = param.data.clone().detach()
                if device is not None:
                    self.ema_values[name] = self.ema_values[name].to(device)

    def update(self):
        """
        Update EMA parameters with current model parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_values

                new_average = self.ema_values[name] - \
                    (1 - self.decay)*(self.ema_values[name] - param.data)
                
                self.ema_values[name] = new_average.clone().detach()

    def apply_ema(self):
        """
        Apply EMA parameters to model for inference.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_values
                self.backup[name] = param.data.clone().detach()
                param.data = self.ema_values[name]

    def restore(self):
        """
        Restore original model parameters after inference.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
