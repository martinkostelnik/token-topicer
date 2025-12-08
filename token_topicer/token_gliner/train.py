import os
from clearml import Task
import lightning as L
from lightning.pytorch.cli import LightningCLI

from model import TokenGlinerModule
from dataset import TokenGlinerDataModule


class TokenGlinerCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        config = self.config.fit

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if config.experiment.project is not None:
            self.task = Task.init(
                project_name=config.experiment.project,
                task_name=config.experiment.name,
            )

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--experiment.project", type=str, default=None)
        parser.add_argument("--experiment.name", type=str)


def cli_main():
    cli = TokenGlinerCLI(
        model_class=TokenGlinerModule,
        datamodule_class=TokenGlinerDataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    cli_main()