from pathlib import Path

import torch
import transformers

from token_topicer.mlp import MLP


class CrossEncoderModel(torch.nn.Module):
    def __init__(
        self,
        lm_path: Path,
        output_projection_layers: int = 2,
    ) -> None:
        super().__init__()
        
        self.language_model = transformers.AutoModel.from_pretrained(lm_path)
        self.output_projection = MLP(
            input_dim=self.language_model.config.hidden_size,
            hidden_dim=self.language_model.config.hidden_size,
            output_dim=self.language_model.config.hidden_size,
            n_layers=output_projection_layers,
            dropout=0.0,
            input_dropout=0.0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        projected_outputs = self.output_projection(outputs.last_hidden_state)
        
        return projected_outputs
