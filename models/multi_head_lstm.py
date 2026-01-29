import torch.nn as nn
import torch

NUM_MODULES = 10
GRID_MODULE_DIM = 2


class MultiHeadLSTM(nn.Module):
    """
    Multi-head LSTM where each head processes independently (no cross-head interactions).
    Uses the inefficient, non-parallel approach as proof-of-concept.
    """

    def __init__(self, input_size, hidden_size, num_heads):
        """
        Parameters are in terms of each head.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.lstms = nn.ModuleList(
            [
                nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
                for i in range(self.num_heads)
            ]
        )

    def forward(
        self,
        input: torch.Tensor,
        initial: torch.Tensor,
        candidate: torch.Tensor,
        input_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert input.shape[-1] % self.input_size == 0, (
            "Input feature must be divisible by per-module input_size"
        )
        seq_outputs = []
        final_hiddens = []
        final_candidates = []
        for i in range(self.num_heads):
            lstm = self.lstms[i]
            per_head_input = input[:, :, i].contiguous()
            # packed = torch.nn.utils.rnn.pack_padded_sequence(
            #     per_head_input, input_length, batch_first=True, enforce_sorted=False
            # )
            seq_hiddens, final_hidden = lstm(
                per_head_input,
                (
                    initial[:, i].contiguous().unsqueeze(0),
                    candidate[:, i].contiguous().unsqueeze(0),
                ),
            )
            seq_outputs.append(seq_hiddens.unsqueeze(1))
            final_hiddens.append(final_hidden[0])
            final_candidates.append(final_hidden[1])
        return torch.concat(seq_outputs, dim=2), (
            torch.concat(final_hiddens, dim=-1),
            torch.concat(final_candidates, dim=-1),
        )
