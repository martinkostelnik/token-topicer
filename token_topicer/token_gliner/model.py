from pathlib import Path

import lightning as L
import torch
import transformers


class TokenGlinerModule(L.LightningModule):
    def __init__(
        self,
        lm_path: Path,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()

        self.lm_path = lm_path
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = TokenGlinerModel(
            lm_path=lm_path,
        )
        self.model.train()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs["mean_loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["input_ids"].shape[0])
        return loss
    
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs["mean_loss"]
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["input_ids"].shape[0])
        return loss

    def common_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        # Training step implementation
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].transpose(1, 2)

        model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        ent_tokens_tensor = self.create_entity_token_tensor(model_outputs, batch["max_entities"], batch["entity_token_indices"])
        word_token_tensor = self.create_word_token_tensor(model_outputs, batch["sep_token_index"])

        scores = torch.bmm(ent_tokens_tensor, word_token_tensor.transpose(1, 2))
        mask = labels != -1
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, reduction="none")
        loss = loss * mask
        loss = loss.sum() / mask.sum()

        return {
            "mean_loss": loss,
        }
    
    def create_entity_token_tensor(
        self,
        model_outputs: torch.Tensor,
        max_entities: int,
        entity_token_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        assert model_outputs.shape[0] == len(entity_token_indices)

        ent_tokens_features = []
        for model_output, ent_indices in zip(model_outputs, entity_token_indices):
            ent_token_features = model_output[ent_indices, :]
            ent_tokens_features.append(ent_token_features)

        padded_ent_tokens = [torch.nn.functional.pad(
            ent_tokens,
            (0, 0, 0, max_entities - ent_tokens.shape[0]),
            value=0.0,
        ) for ent_tokens in ent_tokens_features]

        padded_ent_tokens = torch.stack(padded_ent_tokens)
        return padded_ent_tokens
    
    def create_word_token_tensor(
        self,
        model_outputs: torch.Tensor,
        sep_token_indices: list[int],
    ) -> torch.Tensor:
        assert model_outputs.shape[0] == len(sep_token_indices)

        word_token_features = []
        for model_output, sep_index in zip(model_outputs, sep_token_indices):
            word_token_feature = model_output[sep_index+1:, :]
            word_token_features.append(word_token_feature)

        word_token_tensor = torch.nn.utils.rnn.pad_sequence(
            word_token_features,
            batch_first=True,
            padding_value=0.0,
        )
        return word_token_tensor

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class TokenGlinerModel(torch.nn.Module):
    def __init__(
        self,
        lm_path: Path,
    ) -> None:
        super().__init__()
        
        self.language_model = transformers.AutoModel.from_pretrained(lm_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        return outputs.last_hidden_state
