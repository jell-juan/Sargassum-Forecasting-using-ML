"""
model.py

Device-agnostic ConvLSTM for Sargassum (AFAI) forecasting.

Fixes applied relative to the original snippet:
  - Removed `torch.autograd.Variable` (obsolete since PyTorch 0.4; plain
    tensors carry autograd natively).
  - Removed every internal `.cuda()` / `torch.cuda.is_available()` call.
    A model has no business deciding its own device -- it should simply
    operate on whatever device its own parameters and inputs already live
    on. The caller (train.py) is the only place `device` is decided, via
    `model.to(device)` / `tensor.to(device)`. That's what makes this
    genuinely portable across CUDA, CPU, or any future backend, and avoids
    the model/input silently ending up on different devices.
  - Added an explicit prediction HEAD (a 1x1 Conv2d) mapping the ConvLSTM's
    hidden_dim feature maps down to a single-channel AFAI prediction. The
    original snippet's training loop compared a (batch, hidden_dim, H, W)
    hidden state directly against a (batch, 1, H, W) target -- a shape
    mismatch that would not have run as-is.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # input, forget, output, candidate gates
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device, dtype):
        """Device/dtype are passed in explicitly by the caller (inferred from the actual
        input tensor) rather than decided here -- this is what makes the cell portable."""
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype),
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super().__init__()
        self._check_kernel_size_consistency(kernel_size)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        if not len(hidden_dim) == len(kernel_size) == num_layers:
            raise ValueError("hidden_dim and kernel_size must have the same length as num_layers")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim[i], kernel_size[i], bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        input_tensor: (batch, seq_len, input_dim, H, W) if batch_first, else
                      (seq_len, batch, input_dim, H, W).
        Returns: last_state (h, c) of the LAST layer, and that layer's full
                 output sequence -- unless return_all_layers=True, in which
                 case both are returned as lists across all layers.
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()
        device, dtype = input_tensor.device, input_tensor.dtype

        if hidden_state is None:
            hidden_state = [cell.init_hidden(b, (h, w), device, dtype) for cell in self.cell_list]

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h_state, c_state = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h_state, c_state = self.cell_list[layer_idx](cur_layer_input[:, t], (h_state, c_state))
                output_inner.append(h_state)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append((h_state, c_state))

        if not self.return_all_layers:
            return last_state_list[-1], layer_output_list[-1]
        return last_state_list, layer_output_list

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(isinstance(k, tuple) for k in kernel_size))):
            raise ValueError("kernel_size must be a tuple or a list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        return param if isinstance(param, list) else [param] * num_layers


class SargassumConvLSTM(nn.Module):
    """
    ConvLSTM backbone + a 1x1 Conv2d prediction head, producing a single-
    channel AFAI forecast map from the last timestep's hidden state.

    Input:  (batch, seq_len, in_channels, H, W)
    Output: (batch, 1, H, W)  -- raw (unnormalized) AFAI units, since the
            target is never normalized (only the input channels are).
    """
    def __init__(self, in_channels, hidden_dim=64, kernel_size=(3, 3), num_layers=2):
        super().__init__()
        self.convlstm = ConvLSTM(
            input_dim=in_channels, hidden_dim=hidden_dim, kernel_size=kernel_size,
            num_layers=num_layers, batch_first=True, bias=True, return_all_layers=False,
        )
        final_hidden = hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim
        self.head = nn.Conv2d(final_hidden, 1, kernel_size=1)

    def forward(self, x):
        (h_last, _), _ = self.convlstm(x)
        return self.head(h_last)