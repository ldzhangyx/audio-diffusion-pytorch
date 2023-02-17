import torch


def repeat_padding(input_tensor, required_length):
    """
    Repeat padding to the required length.
    """
    # input_tensor = (Batch x dim x Length)
    # required_length = int
    batch_size, dim, length = input_tensor.shape
    repeat_num = required_length // length
    remainder = required_length % length
    input_tensor = torch.cat([input_tensor] * repeat_num, dim=-1)
    # zero padding
    if remainder > 0:
        input_tensor = torch.cat([input_tensor, input_tensor[:, :, :remainder]], dim=-1)
    return input_tensor


