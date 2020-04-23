import os
import warnings
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from kdtools.datasets import (
    BILUOVSentencesDS,
    DependencyTreeDS,
    SelectedDS,
    from_biluov,
    get_nlp,
    match_tokens_to_entities,
    to_biluov,
)
from kdtools.encoders import SequenceCharEncoder
from kdtools.layers import CharEmbeddingEncoder
from kdtools.models import (
    BasicSequenceClassifier,
    BasicSequenceTagger,
    BertBasedSequenceTagger,
)
from kdtools.nlp import BertNLP
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

    def __init__(
        self,
        taskA_models=None,
        taskB_model=None,
        *,
        only_representative=False,
        bert_mode=None,
        only_bert=False,
    ):
        nlp = get_nlp()
        self.nlp = nlp if bert_mode is None else BertNLP(nlp, merge=bert_mode)
        self.bert_mode = bert_mode
        self.taskA_models: Dict[nn.Module] = taskA_models
        self.taskB_model: nn.Module = taskB_model
        self.only_representative = only_representative
        self.only_bert = only_bert

    def run(self, collection: Collection, *args, taskA: bool, taskB: bool, **kargs):
        if taskA:
            if self.taskA_models is None:
                warnings.warn("No model for taskA available. Skipping ...")
            else:
                self.run_taskA(collection, *args, **kargs)
        if taskB:
            if self.taskB_model is None:
                warnings.warn("No model for taskB available. Skipping ...")
            else:
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

        with torch.no_grad():
            for sid, (*s_features, _) in tqdm(
                enumerate(dataset.shallow_dataloader()),
                total=len(dataset),
                desc=entity_label,
            ):
                tokensxsentence = dataset.tokensxsentence[sid]
                output = model(s_features)
                output = model.decode(output)
                labels = [dataset.labels[x] for x in output]
                decoded = from_biluov(labels, tokensxsentence, spans=True)

                sentence = collection.sentences[sid]
                for spans in decoded:
                    keyphrase = Keyphrase(sentence, entity_label, -1, spans)
                    sentence.keyphrases.append(keyphrase)

    def run_taskB(self, collection: Collection, *args, **kargs):
        model = self.taskB_model

        dataset = self.build_taskB_dataset(collection, inclusion=1.1, predict=True)

        with torch.no_grad():
            for *features, (sid, s_id, d_id) in tqdm(
                dataset.shallow_dataloader(), total=len(dataset), desc="Relations",
            ):
                s_id = s_id.item()
                d_id = d_id.item()

                output = model(features).squeeze(0)
                output = output.argmax(dim=-1)
                label = dataset.labels[output.item()]

                if label is None:
                    continue

                sentence = collection.sentences[sid]
                rel_origin = sentence.keyphrases[s_id].id
                rel_destination = sentence.keyphrases[d_id].id

                relation = Relation(sentence, rel_origin, rel_destination, label)
                sentence.relations.append(relation)

    def train(
        self,
        collection: Collection,
        validation: Collection,
        *,
        jointly,
        inclusion,
        n_epochs=100,
        save_to=None,
        early_stopping=None,
        use_crf=True,
        weight=True,
    ):
        self.train_taskA(
            collection,
            validation,
            jointly,
            n_epochs,
            save_to=save_to,
            early_stopping=early_stopping,
            use_crf=use_crf,
        )
        self.train_taskB(
            collection,
            validation,
            jointly,
            inclusion,
            n_epochs,
            save_to=save_to,
            early_stopping=early_stopping,
            weight=weight,
        )

    def train_taskA(
        self,
        collection: Collection,
        validation: Collection,
        jointly,
        n_epochs=100,
        save_to=None,
        early_stopping=None,
        use_crf=True,
    ):
        if self.only_bert and jointly:
            raise ValueError('Cannot train jointly while using only BERT model!')

        char_encoder = None

        models = {}
        datasets = {}
        validations = {}
        for label in ENTITIES:
            dataset = self.build_taskA_dataset(collection, label)
            model = self.build_taskA_model(
                dataset, n_epochs, shared=char_encoder, use_crf=use_crf
            )
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
                early_stopping=early_stopping,
                use_crf=use_crf,
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
                    early_stopping=early_stopping,
                    use_crf=use_crf,
                )

        self.taskA_models = models

    def build_taskA_model(
        self, dataset: BILUOVSentencesDS, n_epochs=100, *, shared=None, use_crf=True
    ):

        if self.only_bert:
            model = BertBasedSequenceTagger(
                word_repr_dim=dataset.vectors_len,
                num_labels=dataset.label_size,
                use_crf=use_crf,
            )
        else:
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
                use_crf=use_crf,
            )

        return model

    def build_taskA_dataset(self, collection: Collection, entity_label: str):
        sentences = [s.text for s in collection.sentences]
        entities = [
            [k.spans for k in s.keyphrases if k.label == entity_label]
            for s in collection.sentences
        ]
        dataset = BILUOVSentencesDS(sentences, entities, language=self.nlp)
        if self.only_bert:
            dataset = SelectedDS(dataset, 1)
        return dataset

    def train_taskA_model(
        self,
        model,
        dataset,
        validation,
        desc,
        n_epochs=100,
        save_to: str = None,
        early_stopping=None,
        use_crf=True,
    ):
        train_on_shallow_dataloader(
            model,
            dataset,
            validation,
            optim=optim.SGD,
            criterion=model.crf_loss if use_crf else None,
            predictor=model.decode if use_crf else None,
            n_epochs=n_epochs,
            desc=desc,
            save_to=save_to,
            early_stopping=early_stopping,
            extra_config=dict(bert=self.bert_mode),
        )

    def train_all_taskA_models(
        self,
        models,
        datasets,
        validations,
        desc,
        n_epochs=100,
        save_to: str = None,
        early_stopping=None,
        use_crf=True,
    ):
        jointly_train_on_shallow_dataloader(
            models,
            datasets,
            validations,
            optim=optim.SGD,
            criterion=(lambda model: model.crf_loss) if use_crf else None,
            predictor=(lambda model: model.decode) if use_crf else None,
            n_epochs=n_epochs,
            desc=desc,
            save_to=save_to,
            early_stopping=early_stopping,
            extra_config=dict(bert=self.bert_mode),
        )

    def train_taskB(
        self,
        collection: Collection,
        validation: Collection,
        jointly,
        inclusion,
        n_epochs=100,
        save_to=None,
        early_stopping=None,
        weight=True,
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

        criterion = nn.CrossEntropyLoss(weight=dataset.weights()) if weight else None
        validation_criterion = (
            nn.CrossEntropyLoss(weight=validation_ds.weights()) if weight else None
        )

        train_on_shallow_dataloader(
            model,
            dataset,
            validation_ds,
            criterion=criterion,
            validation_criterion=validation_criterion,
            n_epochs=n_epochs,
            desc="relations",
            save_to=save_to("taskB"),
            early_stopping=early_stopping,
            extra_config=dict(bert=self.bert_mode),
        )

        self.taskB_model = model

    def build_taskB_dataset(self, collection: Collection, inclusion, predict=False):
        tokensxsentence = [self.nlp(s.text) for s in collection.sentences]
        entities = [
            [(k.spans, k.label) for k in s.keyphrases] for s in collection.sentences
        ]

        entitiesxsentence, token2label = match_tokens_to_entities(
            tokensxsentence, entities, only_representative=self.only_representative
        )

        keyphrase2tokens = {
            keyphrase: token
            for sentence, tokens in zip(collection.sentences, entitiesxsentence)
            for keyphrase, token in zip(sentence.keyphrases, tokens)
        }
        relations = (
            {
                (
                    keyphrase2tokens[rel.from_phrase],
                    keyphrase2tokens[rel.to_phrase],
                ): rel.label
                for sentence in collection.sentences
                for rel in sentence.relations
            }
            if not predict
            else None
        )

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

    def _training_task(
        n_epochs,
        *,
        bert_mode,
        inclusion=0.1,
        task=None,
        jointly=True,
        early_stopping=None,
        use_crf=True,
        weight=True,
        only_bert=False,
    ):
        training = Collection().load(Path("data/training/scenario.txt"))
        validation = Collection().load(Path("data/development/main/scenario.txt"))

        early_stopping = early_stopping or dict(wait=5, delta=0.0)

        algorithm = UHMajaModel(bert_mode=bert_mode, only_bert=only_bert)
        if task is None:
            algorithm.train(
                training,
                validation,
                jointly=jointly,
                inclusion=inclusion,
                n_epochs=n_epochs,
                save_to=name_to_path,
                early_stopping=early_stopping,
                use_crf=use_crf,
                weight=weight,
            )
        elif task == "A":
            algorithm.train_taskA(
                training,
                validation,
                jointly=jointly,
                n_epochs=n_epochs,
                save_to=name_to_path,
                early_stopping=early_stopping,
                use_crf=use_crf,
            )
        elif task == "B":
            # load A
            if jointly:
                taskA_models = {}
                for label in ENTITIES:
                    checkpoint = torch.load(f"trained/taskA-{label}.pt")
                    _ensure_bert(bert_mode, checkpoint)
                    model = checkpoint["model"]
                    taskA_models[label] = model
                    model.eval()
                algorithm.taskA_models = taskA_models

            algorithm.train_taskB(
                training,
                validation,
                jointly=jointly,
                inclusion=inclusion,
                n_epochs=n_epochs,
                save_to=name_to_path,
                early_stopping=early_stopping,
                weight=weight,
            )

    def _log_checkpoint(checkpoint, *, desc):
        print(f"[{desc}]:".center(80, ":"))
        for key, value in checkpoint.items():
            print(f"{key}: {value}")

    def _ensure_bert(bert_mode, checkpoint):
        try:
            bert = checkpoint["bert"]
            if bert_mode != bert:
                raise ValueError(
                    "The model was not trained using the same configuration of BERT."
                )
        except KeyError:
            if bert_mode is not None:
                raise ValueError("The model was not trained using BERT.")

    def _run_task(*, bert_mode, task=None):
        if task == "B":
            taskA_models = None
        else:
            taskA_models = {}
            for label in ENTITIES:
                checkpoint = torch.load(f"trained/taskA-{label}.pt")
                _log_checkpoint(checkpoint, desc=label)
                _ensure_bert(bert_mode, checkpoint)
                model = checkpoint["model"]
                taskA_models[label] = model
                model.eval()

        if task == "A":
            taskB_model = None
        else:
            checkpoint = torch.load("./trained/taskB.pt")
            _log_checkpoint(checkpoint, desc="Relations")
            _ensure_bert(bert_mode, checkpoint)
            taskB_model = checkpoint["model"]
            taskB_model.eval()

        algorithm = UHMajaModel(taskA_models, taskB_model, bert_mode=bert_mode)

        tasks = handle_args()
        Run.submit("ehealth19-maja", tasks, algorithm)

    def _test_biluov_task():
        import es_core_news_md
        from scripts.utils import Sentence

        def forward(tokensxsentence, entitiesxsentence):
            labelsxsentence, _ = to_biluov(tokensxsentence, entitiesxsentence)
            return [
                from_biluov(biluov, sentence, spans=True)
                for biluov, sentence in zip(labelsxsentence, tokensxsentence)
            ]

        training = Collection().load(Path("data/training/scenario.txt"))
        nlp = es_core_news_md.load()

        def per_label(label):
            tokensxsentence = [nlp(s.text) for s in training.sentences]
            entitiesxsentence = [
                [k.spans for k in s.keyphrases if k.label == label]
                for s in training.sentences
            ]
            decoded = forward(tokensxsentence, entitiesxsentence)
            return decoded

        collection = Collection([Sentence(s.text) for s in training.sentences])
        for label in ENTITIES:
            decoded = per_label(label)
            for entities, sentence in zip(decoded, collection.sentences):
                for spans in entities:
                    keyphrase = Keyphrase(sentence, label, -1, spans)
                    sentence.keyphrases.append(keyphrase)

        collection.fix_ids()
        output = Path("data/submissions/forward-biluov/train/run1/scenario2-taskA/")
        output.mkdir(parents=True, exist_ok=True)
        collection.dump(output / "scenario.txt", skip_empty_sentences=False)
