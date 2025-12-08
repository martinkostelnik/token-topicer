import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        input_dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        self.input_dropout = None
        if input_dropout > 0.0:
            self.input_dropout = torch.nn.Dropout(input_dropout)

        self.layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, hidden_dim if n_layers > 1 else output_dim)])

        for i in range(n_layers - 1):
            self.layers.extend([torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim, hidden_dim if i < n_layers - 2 else output_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dropout is not None:
            x = self.input_dropout(x)

        for layer in self.layers:
            x = layer(x)
        
        return x
