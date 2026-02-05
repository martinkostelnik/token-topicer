from safe_gpu import safe_gpu
safe_gpu.claim_gpus()

import os
from pathlib import Path

from clearml import Task
import lightning as L
from lightning.pytorch.cli import LightningCLI
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from omegaconf import OmegaConf

from token_topicer.cross_encoder.dataset import CrossEncoderTopicClassifierDataModule
from token_topicer.cross_encoder.model import CrossEncoderModel 


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0

    def update(self, values: list[float], count: int | None = None):
        self.val += sum(values)
        self.count += len(values) if count is None else count

    def __call__(self):
        value = self.val / self.count if self.count != 0 else 0.0
        self.reset()
        return value


def cross_dot_product(
    model_outputs: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    normalize_score: bool = True,
):
    """
    Computes a similarity matrix between topic tokens and text tokens.
    Handles padding correctly and ignores CLS/SEP tokens.
    
    Args:
        model_outputs: (B, seq_len, hidden_dim) output of LM
        token_type_ids: (B, seq_len), 0=topic, 1=text
        attention_mask: (B, seq_len)
    
    Returns:
        similarity_matrix: (B, topic_len, text_len)
    """
    # Masks for topic and text tokens
    topic_mask = (token_type_ids == 0) & attention_mask.bool()
    text_mask = (token_type_ids == 1) & attention_mask.bool()

    # Extract tokens per batch item
    topic_tokens_list = [out[m.bool()][1:-1] for out, m in zip(model_outputs, topic_mask)]  # exclude CLS and SEP
    text_tokens_list = [out[m.bool()][:-1] for out, m in zip(model_outputs, text_mask)]     # exclude SEP

    # Pad sequences to max length in batch
    topic_tokens = torch.nn.utils.rnn.pad_sequence(topic_tokens_list, batch_first=True)
    text_tokens = torch.nn.utils.rnn.pad_sequence(text_tokens_list, batch_first=True)

    # Create masks for padded tokens
    topic_lens = torch.tensor([t.shape[0] for t in topic_tokens_list], device=model_outputs.device)
    text_lens = torch.tensor([t.shape[0] for t in text_tokens_list], device=model_outputs.device)

    text_pad_mask = (torch.arange(text_tokens.size(1), device=model_outputs.device)[None, :] < text_lens[:, None])
    top_pad_mask = (torch.arange(topic_tokens.size(1), device=model_outputs.device)[None, :] < topic_lens[:, None])

    # Compute similarity
    similarity_matrix = torch.bmm(topic_tokens, text_tokens.transpose(1, 2))  # (B, topic_len, text_len)
    if normalize_score:
        similarity_matrix = similarity_matrix / torch.sqrt(torch.tensor(model_outputs.size(-1), dtype=torch.float32, device=model_outputs.device))

    # Mask out padded text positions with -inf so max ignores them
    similarity_matrix = similarity_matrix.masked_fill(~text_pad_mask[:, None, :], -1e10)
    similarity_matrix = similarity_matrix.masked_fill(~top_pad_mask[:, :, None], -1e10)

    return similarity_matrix

class CrossEncoderTopicClassifierModule(L.LightningModule):
    def __init__(
        self,
        lm_path: Path,
        learning_rate: float,
        weight_decay: float,
        output_projection_layers: int = 2,
        scale_positives: bool = True,
        normalize_score: bool = True,
        soft_max_score: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lm_path = lm_path
        self.learning_rate = learning_rate
        self.scale_positives = scale_positives
        self.weight_decay = weight_decay
        self.normalize_score = normalize_score
        self.soft_max_score = soft_max_score

        self.model = CrossEncoderModel(
            lm_path=lm_path,
            output_projection_layers=output_projection_layers,
        )
        self.model.train()

        print(self.model)

        self.train_loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()

        self.running_train_labels = []
        self.running_val_labels = []
        self.running_train_scores = []
        self.running_val_scores = []

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs["loss"]
        scores = outputs["scores"]
        mask = outputs["mask"]
        self.train_loss_meter.update(loss.detach().cpu().flatten().tolist(), mask.sum().item())

        self.running_train_labels.extend(batch["labels"][mask].detach().cpu().flatten().tolist())
        self.running_train_scores.extend(scores[mask].detach().cpu().flatten().tolist())

        return loss.sum() / mask.sum()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.log("loss/train", self.train_loss_meter(), on_step=True, on_epoch=False, prog_bar=True)
            if len(self.running_train_labels) > 1:
                train_auc = roc_auc_score(self.running_train_labels, self.running_train_scores)
                train_ap = average_precision_score(self.running_train_labels, self.running_train_scores)
                self.log("auc/train", train_auc, on_step=True, on_epoch=False, prog_bar=True)
                self.log("ap/train", train_ap, on_step=True, on_epoch=False, prog_bar=True)
                self.running_train_labels = []
                self.running_train_scores = []

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs["loss"]
        scores = outputs["scores"]
        mask = outputs["mask"]
        self.val_loss_meter.update(loss.detach().cpu().flatten().tolist(), mask.sum().item())
        self.running_val_labels.extend(batch["labels"][mask].detach().cpu().flatten().tolist())
        self.running_val_scores.extend(scores[mask].detach().cpu().flatten().tolist())
        return loss.sum() / mask.sum()

    def on_validation_epoch_end(self):
        meter = self.val_loss_meter
        labels = self.running_val_labels
        scores = self.running_val_scores

        self.log("loss/val", meter(), prog_bar=True)
        if len(labels) > 1:
            val_auc = roc_auc_score(labels, scores)
            val_ap = average_precision_score(labels, scores)
            self.log("ap/val", val_ap, prog_bar=True)
            self.log("auc/val", val_auc, prog_bar=True)
            self.running_val_labels = []
            self.running_val_scores = []

    def common_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]

        # Model forward
        model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get similarity matrix and text padding mask
        similarity_matrix = cross_dot_product(
            model_outputs=model_outputs,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            normalize_score=self.normalize_score,
        )

        # Max over topic tokens
        if self.soft_max_score:
            scores = torch.logsumexp(similarity_matrix, dim=1)  # shape (B, S_text)
        else:
            scores = similarity_matrix.max(dim=1).values  # shape (B, S_text)

        # Compute BCE loss with masking
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores,
            labels,
            reduction="none",
            pos_weight=batch["pos_weight"] if self.scale_positives else None,
        )
        mask = labels != -1
        loss = loss * mask.float()

        return {
            "loss": loss,
            "scores": scores,
            "mask": mask,
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
    

class TokenGlinerCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        config = self.config.fit

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if config.experiment.project is not None:
            self.clearml_task = Task.init(
                project_name=config.experiment.project,
                task_name=config.experiment.name,
            )

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--experiment.project", type=str, default=None)
        parser.add_argument("--experiment.name", type=str)


def cli_main():
    OmegaConf.register_new_resolver("replace_slash", lambda s: s.replace("/", "_"))
    cli = TokenGlinerCLI(
        model_class=CrossEncoderTopicClassifierModule,
        datamodule_class=CrossEncoderTopicClassifierDataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    cli_main()
