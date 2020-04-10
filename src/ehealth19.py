import os
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from kdtools.datasets import (
    BILUOVSentencesDS,
    DependencyTreeDS,
    from_biluov,
    get_nlp,
    match_tokens_to_entities,
)
from kdtools.encoders import SequenceCharEncoder
from kdtools.layers import CharEmbeddingEncoder
from kdtools.models import BasicSequenceClassifier, BasicSequenceTagger
from kdtools.utils import (
    jointly_train_on_shallow_dataloader,
    train_on_shallow_dataloader,
)
from tqdm import tqdm

from scripts.submit import Algorithm, Run, handle_args
from scripts.utils import ENTITIES, RELATIONS, Collection, Keyphrase, Relation


class UHMajaModel(Algorithm):
    CHAR_EMBEDDING_DIM = 100
    CHAR_REPR_DIM = 200
    TOKEN_REPR_DIM = 300

    def __init__(self, taskA_models=None, taskB_model=None):
        self.nlp = get_nlp()
        self.taskA_models: Dict[nn.Module] = taskA_models
        self.taskB_model: nn.Module = taskB_model

    def run(self, collection: Collection, *args, taskA: bool, taskB: bool, **kargs):
        if taskA:
            self.run_taskA(collection, *args, **kargs)
        if taskB:
            self.run_taskB(collection, *args, **kargs)
        return collection

    def run_taskA(self, collection: Collection, *args, **kargs):
        for label in ENTITIES:
            self.run_taskA_for_label(collection, label, *args, **kargs)
        collection.fix_ids()

    def run_taskA_for_label(
        self, collection: Collection, entity_label: str, *args, **kargs
    ):
        model = self.taskA_models[entity_label]
        dataset = BILUOVSentencesDS(
            [s.text for s in collection.sentences], language=self.nlp
        )

        for sid, (*s_features, _) in tqdm(
            enumerate(dataset.shallow_dataloader()),
            total=len(dataset),
            desc=entity_label,
        ):
            tokensxsentence = dataset.tokensxsentence[sid]
            output = model(s_features).squeeze(0)
            output = output.argmax(dim=-1)
            labels = [dataset.labels[x] for x in output]
            decoded = from_biluov(labels, tokensxsentence, spans=True)

            sentence = collection.sentences[sid]
            for spans in decoded:
                keyphrase = Keyphrase(sentence, entity_label, -1, spans)
                sentence.keyphrases.append(keyphrase)

    def run_taskB(self, collection: Collection, *args, **kargs):
        pass

    def train(
        self,
        collection: Collection,
        validation: Collection,
        *,
        jointly,
        inclusion,
        n_epochs=100,
        save_to=None,
    ):
        self.train_taskA(collection, validation, jointly, n_epochs, save_to)
        self.train_taskB(collection, validation, jointly, inclusion, n_epochs, save_to)

    def train_taskA(
        self,
        collection: Collection,
        validation: Collection,
        jointly,
        n_epochs=100,
        save_to=None,
    ):
        char_encoder = None

        models = {}
        datasets = {}
        validations = {}
        for label in ENTITIES:
            dataset = self.build_taskA_dataset(collection, label)
            model = self.build_taskA_model(dataset, n_epochs, shared=char_encoder)
            validation_ds = self.build_taskA_dataset(validation, label)

            models[label] = model
            datasets[label] = dataset
            validations[label] = validation_ds

            char_encoder = model.char_encoder if jointly else None

        if jointly:
            # dicts are stable
            self.train_all_taskA_models(
                models.values(),
                datasets.values(),
                validations.values(),
                "all",
                n_epochs,
                save_to=(
                    [save_to(label) for label in ENTITIES]
                    if save_to is not None
                    else None
                ),
            )
        else:
            for label in ENTITIES:
                self.train_taskA_model(
                    models[label],
                    datasets[label],
                    validations[label],
                    label,
                    n_epochs,
                    save_to=save_to(label) if save_to is not None else None,
                )

        self.taskA_models = models

    def build_taskA_model(self, dataset: BILUOVSentencesDS, n_epochs=100, shared=None):
        model = BasicSequenceTagger(
            char_vocab_size=dataset.char_size,
            char_embedding_dim=self.CHAR_EMBEDDING_DIM,
            padding_idx=dataset.padding,
            char_repr_dim=self.CHAR_REPR_DIM,
            word_repr_dim=dataset.vectors_len,
            postag_repr_dim=dataset.pos_size,
            token_repr_dim=self.TOKEN_REPR_DIM,
            num_labels=dataset.label_size,
            char_encoder=shared,
        )

        return model

    def build_taskA_dataset(self, collection: Collection, entity_label: str):
        sentences = [s.text for s in collection.sentences]
        entities = [
            [k.spans for k in s.keyphrases if k.label == entity_label]
            for s in collection.sentences
        ]
        dataset = BILUOVSentencesDS(sentences, entities, language=self.nlp)
        return dataset

    def train_taskA_model(
        self, model, dataset, validation, desc, n_epochs=100, save_to: str = None
    ):
        train_on_shallow_dataloader(
            model,
            dataset,
            validation,
            optim=optim.SGD,
            criterion=nn.CrossEntropyLoss,
            n_epochs=n_epochs,
            desc=desc,
            save_to=save_to,
        )

    def train_all_taskA_models(
        self, models, datasets, validations, desc, n_epochs=100, save_to: str = None
    ):
        jointly_train_on_shallow_dataloader(
            models,
            datasets,
            validations,
            optim=optim.SGD,
            criterion=nn.CrossEntropyLoss,
            n_epochs=n_epochs,
            desc=desc,
            save_to=save_to,
        )

    def train_taskB(
        self,
        collection: Collection,
        validation: Collection,
        jointly,
        inclusion,
        n_epochs=100,
        save_to=None,
    ):

        dataset = self.build_taskB_dataset(collection, inclusion)
        char2repr = (
            next(iter(self.taskA_models.values())).char_encoder if jointly else None
        )

        model = BasicSequenceClassifier(
            char_vocab_size=dataset.char_size,
            char_embedding_dim=self.CHAR_EMBEDDING_DIM,
            padding_idx=dataset.padding,
            char_repr_dim=self.CHAR_REPR_DIM,
            word_repr_dim=dataset.vectors_len,
            postag_repr_dim=dataset.pos_size,
            dep_repr_dim=dataset.dep_size,
            entity_repr_dim=dataset.ent_size,
            subtree_repr_dim=self.TOKEN_REPR_DIM,
            token_repr_dim=self.TOKEN_REPR_DIM,
            num_labels=dataset.label_size,
            char_encoder=char2repr,
            already_encoded=False,
            freeze=True,
        )

        validation_ds = self.build_taskB_dataset(validation, inclusion=1.1)

        train_on_shallow_dataloader(
            model,
            dataset,
            validation_ds,
            n_epochs=n_epochs,
            desc="relations",
            save_to=save_to("taskB"),
        )

        self.taskB_model = model

    def build_taskB_dataset(self, collection: Collection, inclusion):
        tokensxsentence = [self.nlp(s.text) for s in collection.sentences]
        entities = [
            [(k.spans, k.label) for k in s.keyphrases] for s in collection.sentences
        ]

        entitiesxsentence, token2label = match_tokens_to_entities(
            tokensxsentence, entities, only_representative=False
        )

        keyphrase2tokens = {
            keyphrase: token
            for sentence, tokens in zip(collection.sentences, entitiesxsentence)
            for keyphrase, token in zip(sentence.keyphrases, tokens)
        }
        relations = {
            (
                keyphrase2tokens[rel.from_phrase],
                keyphrase2tokens[rel.to_phrase],
            ): rel.label
            for sentence in collection.sentences
            for rel in sentence.relations
        }

        dataset = DependencyTreeDS(
            entitiesxsentence,
            relations,
            token2label,
            ENTITIES,
            RELATIONS,
            self.nlp,
            inclusion=inclusion,
            char2repr=None,
        )

        return dataset

    def save_models(self, path="./trained/"):
        for label, model in self.taskA_models.items():
            torch.save(model, os.path.join(path, f"taskA-{label}.pt"))
        torch.save(self.taskB_model, os.path.join(path, "taskB.pt"))


if __name__ == "__main__":
    from pathlib import Path

    def name_to_path(name):
        if name in ENTITIES:
            return f"trained/taskA-{name}.pt"
        if name == "taskB":
            return "trained/taskB.pt"
        raise ValueError("Cannot handle `name`")

    def _training_task():
        training = Collection().load(Path("data/training/scenario.txt"))
        validation = Collection().load(Path("data/development/main/scenario.txt"))

        algorithm = UHMajaModel()
        algorithm.train(
            training,
            validation,
            jointly=True,
            inclusion=0.1,
            n_epochs=1,
            save_to=name_to_path,
        )

    def _run_task():
        taskA_models = {}
        for label in ENTITIES:
            model = torch.load(f"trained/taskA-{label}.pt")
            taskA_models[label] = model
            model.eval()
        taskB_model = torch.load("./trained/taskB.pt")
        taskB_model.eval()
        algorithm = UHMajaModel(taskA_models, taskB_model)

        tasks = handle_args()
        Run.submit("ehealth19-maja", tasks, algorithm)
