import os
import warnings
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from kdtools.datasets import (
    BILUOVSentencesDS,
    DependencyTreeDS,
    FocusOnEntityDS,
    SelectedDS,
    from_biluov,
    match_tokens_to_entities,
    to_biluov,
)
from kdtools.encoders import SequenceCharEncoder
from kdtools.layers import CharEmbeddingEncoder
from kdtools.models import (
    AttentionSequenceTagger,
    BasicSequenceClassifier,
    BasicSequenceTagger,
    BertBasedSequenceClassifier,
    BertBasedSequenceTagger,
)
from kdtools.nlp import BertNLP, get_nlp
from kdtools.utils import (
    jointly_train_on_shallow_dataloader,
    train_on_shallow_dataloader,
)
from tqdm import tqdm

from scripts.submit import Algorithm, Run, handle_args
from scripts.utils import ENTITIES, RELATIONS, Collection, Keyphrase, Relation

TAXONOMIC_RELS = [
    "is-a",
    "same-as",
    "part-of",
    "has-property",
    "causes",
    "entails",
]

CONTEXT_RELS = [
    "in-context",
    "in-place",
    "in-time",
    "subject",
    "target",
    "domain",
    "arg",
]

assert set(TAXONOMIC_RELS + CONTEXT_RELS) == set(RELATIONS)


class eHealth20Model(Algorithm):
    CHAR_EMBEDDING_DIM = 100
    CHAR_REPR_DIM = 200
    TOKEN_REPR_DIM = 300
    POSITIONAL_EMBEDDING_DIM = 100

    def __init__(
        self,
        taskA_models=None,
        taskB_pair_model=None,
        taskB_seq_model=None,
        *,
        only_representative=False,
        bert_mode=None,
        only_bert=False,
        cnet_mode=None,
        tag=None,
    ):
        if only_bert and bert_mode is None:
            raise ValueError("BERT mode not set!")

        nlp = get_nlp()
        self.nlp = nlp if bert_mode is None else BertNLP(nlp, merge=bert_mode)
        self.bert_mode = bert_mode
        self.taskA_models: Dict[nn.Module] = taskA_models
        self.taskB_pair_model: nn.Module = taskB_pair_model
        self.taskB_seq_model: nn.Module = taskB_seq_model
        self.only_representative = only_representative
        self.only_bert = only_bert
        self.cnet_mode = cnet_mode
        self.tag = tag

    def run(self, collection: Collection, *args, taskA: bool, taskB: bool, **kargs):
        if taskA:
            if self.taskA_models is None:
                warnings.warn("No model for taskA available. Skipping ...")
            else:
                self.run_taskA(collection, *args, **kargs)
        if taskB:
            if self.taskB_pair_model is None and self.taskB_seq_model is None:
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
        train_pairs, train_seq = (
            (TAXONOMIC_RELS, CONTEXT_RELS)
            if self.taskB_pair_model is not None and self.taskB_seq_model is not None
            else (RELATIONS, None)
            if self.taskB_pair_model is not None
            else (None, RELATIONS)
            if self.taskB_seq_model is not None
            else (None, None)
        )

        pair_dataset, seq_dataset = self.build_taskB_dataset(
            collection,
            inclusion=1.1,
            predict=True,
            tag=self.tag,
            train_pairs=train_pairs,
            train_seqs=train_seq,
        )

        self.run_taskB_on_pairs(pair_dataset, collection, *args, **kargs)
        self.run_taskB_on_seqs(seq_dataset, collection, *args, **kargs)

    def run_taskB_on_pairs(self, dataset, collection: Collection, *args, **kargs):
        model = self.taskB_pair_model
        if model is None:
            return

        with torch.no_grad():
            for *features, (sid, s_id, d_id) in tqdm(
                dataset.shallow_dataloader(),
                total=len(dataset),
                desc="Relations (Pairs)",
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

    def run_taskB_on_seqs(self, dataset, collection: Collection, *args, **kargs):
        model = self.taskB_seq_model
        if model is None:
            return

        with torch.no_grad():
            for features, i, (sid, head_id, tokens_ids) in tqdm(
                dataset.shallow_dataloader(),
                total=len(dataset),
                desc="Relations (Sequence)",
            ):
                output = model((features, i))
                output = model.decode(output)
                labels = [dataset.labels[x] for x in output]

                sentence = collection.sentences[sid]
                head_entity = sentence.keyphrases[head_id]
                for token_id, label in zip(tokens_ids, labels):
                    if label is None or token_id < 0:
                        continue

                    token_entity = sentence.keyphrases[token_id]

                    rel_origin = head_entity.id
                    rel_destination = token_entity.id
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
        train_pairs=TAXONOMIC_RELS,
        train_seqs=CONTEXT_RELS,
    ):
        self.train_taskA(
            collection,
            validation,
            jointly,
            n_epochs,
            save_to=save_to,
            early_stopping=early_stopping,
            use_crf=use_crf,
            weight=weight,
        )
        self.train_taskB(
            collection,
            validation,
            jointly,
            inclusion,
            n_epochs,
            save_to=save_to,
            early_stopping=early_stopping,
            use_crf=use_crf,
            weight=weight,
            train_pairs=train_pairs,
            train_seqs=train_seqs,
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
        weight=True,
    ):
        if self.only_bert and jointly:
            raise ValueError("Cannot train jointly while using only BERT model!")

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
                weight=weight,
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
                    weight=weight,
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
        weight=True,
    ):
        if use_crf and weight:
            warnings.warn(
                "Using both CRF and weighting in taskA model. `weight` will be ignored."
            )

        criterion = (
            model.crf_loss
            if use_crf
            else nn.CrossEntropyLoss(weight=dataset.weights())
            if weight
            else None
        )
        validation_criterion = (
            model.crf_loss
            if use_crf
            else nn.CrossEntropyLoss(weight=validation.weights())
            if weight
            else None
        )

        train_on_shallow_dataloader(
            model,
            dataset,
            validation,
            optim=optim.SGD,
            criterion=criterion,
            validation_criterion=validation_criterion,
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
        weight=True,
    ):
        if use_crf and weight:
            warnings.warn(
                "Using both CRF and weighting in taskA model. `weight` will be ignored."
            )

        if use_crf:
            criterion = lambda i, model: model.crf_loss
            validation_criterion = None
        elif weight:
            _criterion = [
                nn.CrossEntropyLoss(weight=dataset.weights()) for dataset in datasets
            ]
            criterion = lambda i, model: _criterion[i]
            _validation_criterion = [
                nn.CrossEntropyLoss(weight=validation.weights())
                for validation in validations
            ]
            validation_criterion = lambda i, model: _validation_criterion[i]
        else:
            criterion, validation_criterion = None, None

        jointly_train_on_shallow_dataloader(
            models,
            datasets,
            validations,
            optim=optim.SGD,
            criterion=criterion,
            validation_criterion=validation_criterion,
            predictor=(lambda i, model: model.decode) if use_crf else None,
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
        use_crf=True,
        train_pairs=TAXONOMIC_RELS,
        train_seqs=CONTEXT_RELS,
    ):
        if weight and inclusion <= 1:
            warnings.warn(
                "Since using `weight=True`, you probably meant to set `inclusion=1.1`."
            )
        if self.only_bert and jointly:
            raise ValueError("Cannot train jointly while using only BERT model!")

        print(f"Training pairs: {train_pairs}")
        print(f"Training seqs: {train_seqs}")

        dataset1, dataset2 = self.build_taskB_dataset(
            collection,
            inclusion,
            tag="train",
            train_pairs=train_pairs,
            train_seqs=train_seqs,
        )
        validation_ds1, validation_ds2 = self.build_taskB_dataset(
            validation,
            inclusion=1.1,
            tag="dev",
            train_pairs=train_pairs,
            train_seqs=train_seqs,
        )

        char2repr = (
            next(iter(self.taskA_models.values())).char_encoder if jointly else None
        )

        if dataset1 is not None:

            if self.only_bert:
                model = BertBasedSequenceClassifier(
                    word_repr_dim=dataset1.vectors_len,
                    num_labels=dataset1.label_size,
                    merge_mode=self.bert_mode,
                )
            else:
                model = BasicSequenceClassifier(
                    char_vocab_size=dataset1.char_size,
                    char_embedding_dim=self.CHAR_EMBEDDING_DIM,
                    padding_idx=dataset1.padding,
                    char_repr_dim=self.CHAR_REPR_DIM,
                    word_repr_dim=dataset1.vectors_len,
                    postag_repr_dim=dataset1.pos_size,
                    dep_repr_dim=dataset1.dep_size,
                    entity_repr_dim=dataset1.ent_size,
                    subtree_repr_dim=self.TOKEN_REPR_DIM,
                    token_repr_dim=self.TOKEN_REPR_DIM,
                    num_labels=dataset1.label_size,
                    char_encoder=char2repr,
                    already_encoded=False,
                    freeze=True,
                    pairwise_info_size=dataset1.pair_size,
                )
            criterion = (
                nn.CrossEntropyLoss(weight=dataset1.weights()) if weight else None
            )
            validation_criterion = (
                nn.CrossEntropyLoss(weight=validation_ds1.weights()) if weight else None
            )

            train_on_shallow_dataloader(
                model,
                dataset1,
                validation_ds1,
                criterion=criterion,
                validation_criterion=validation_criterion,
                n_epochs=n_epochs,
                desc="relations (pairs)",
                save_to=save_to("taskB-pairs"),
                early_stopping=early_stopping,
                extra_config=dict(bert=self.bert_mode, cnet=self.cnet_mode),
            )

            self.taskB_pair_model = model

        if dataset2 is not None:

            ## THIS IS NOT CONVENIENT
            # char2repr = (
            #     self.taskB_pair_model.char_encoder
            #     if jointly and char2repr is None and self.taskB_pair_model is not None
            #     else char2repr
            # )

            model = AttentionSequenceTagger(
                char_vocab_size=dataset2.char_size,
                char_embedding_dim=self.CHAR_EMBEDDING_DIM,
                padding_idx=dataset2.padding,
                char_repr_dim=self.CHAR_REPR_DIM,
                word_repr_dim=dataset2.vectors_len,
                postag_repr_dim=dataset2.pos_size,
                dep_repr_dim=dataset2.dep_size,
                entity_repr_dim=dataset2.ent_size,
                token_repr_dim=self.TOKEN_REPR_DIM,
                position_repr_dim=dataset2.positional_size,
                num_labels=dataset2.label_size,
                char_encoder=char2repr,
                already_encoded=False,
                freeze=True,
                use_crf=use_crf,
                pairwise_repr_dim=dataset2.pair_size,
            )

            if use_crf and weight:
                warnings.warn(
                    "Using both CRF and weighting in sequence relation model. `weight` will be ignored."
                )

            criterion = (
                model.crf_loss
                if use_crf
                else nn.CrossEntropyLoss(weight=dataset2.weights())
                if weight
                else None
            )
            validation_criterion = (
                model.crf_loss
                if use_crf
                else nn.CrossEntropyLoss(weight=validation_ds2.weights())
                if weight
                else None
            )
            predictor = model.decode if use_crf else None

            train_on_shallow_dataloader(
                model,
                dataset2,
                validation_ds2,
                criterion=criterion,
                validation_criterion=validation_criterion,
                predictor=predictor,
                n_epochs=n_epochs,
                desc="relations (sequence)",
                save_to=save_to("taskB-seqs"),
                early_stopping=early_stopping,
                extra_config=dict(bert=self.bert_mode, cnet=self.cnet_mode),
            )

            self.taskB_seq_model = model

    def build_taskB_dataset(
        self,
        collection: Collection,
        inclusion,
        predict=False,
        tag=None,
        train_pairs=TAXONOMIC_RELS,
        train_seqs=CONTEXT_RELS,
    ):
        if train_pairs is None and train_seqs is None:
            return None, None

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

        def get_relations(labels):
            return (
                {
                    (
                        keyphrase2tokens[rel.from_phrase],
                        keyphrase2tokens[rel.to_phrase],
                    ): rel.label
                    for sentence in collection.sentences
                    for rel in sentence.relations
                    if rel.label in labels
                }
                if not predict
                else None
            )

        pair_dataset = (
            DependencyTreeDS(
                entitiesxsentence,
                get_relations(set(train_pairs)),
                token2label,
                ENTITIES,
                train_pairs,
                self.nlp,
                inclusion=inclusion,
                char2repr=None,
                conceptnet=(
                    self.cnet_mode
                    if tag is None
                    else {"mode": self.cnet_mode, "tag": tag}
                ),
            )
            if train_pairs is not None
            else None
        )

        seq_dataset = (
            FocusOnEntityDS(
                tokensxsentence,
                entitiesxsentence,
                get_relations(set(train_seqs)),
                token2label,
                ENTITIES,
                train_seqs,
                self.nlp,
                self.POSITIONAL_EMBEDDING_DIM,
                char2repr=None,
                conceptnet=(
                    self.cnet_mode
                    if tag is None
                    else {"mode": self.cnet_mode, "tag": tag}
                ),
            )
            if train_seqs is not None
            else None
        )

        return pair_dataset, seq_dataset

    def save_models(self, path="./trained/"):
        for label, model in self.taskA_models.items():
            torch.save(model, os.path.join(path, f"taskA-{label}.pt"))
        torch.save(self.taskB_pair_model, os.path.join(path, "taskB.pt"))
        torch.save(self.taskB_seq_model, os.path.join(path, "taskB-seqs.pt"))


if __name__ == "__main__":
    from pathlib import Path

    def name_to_path(name):
        if name in ENTITIES:
            return f"trained/taskA-{name}.pt"
        if name == "taskB-pairs":
            return "trained/taskB.pt"
        if name == "taskB-seqs":
            return "trained/taskB-seqs.pt"
        raise ValueError("Cannot handle `name`")

    def _training_task(
        n_epochs,
        *,
        bert_mode,
        cnet_mode,
        inclusion=1.1,
        task=None,
        jointly=True,
        early_stopping=None,
        use_crf=True,
        weight=True,
        only_bert=False,
        split_relations="both",
    ):
        if split_relations not in ("both", "pair", "seq"):
            raise ValueError()

        training = Collection().load(Path("data/training/scenario.txt"))
        validation = Collection().load(Path("data/development/main/scenario.txt"))

        early_stopping = early_stopping or dict(wait=5, delta=0.0)

        algorithm = eHealth20Model(
            bert_mode=bert_mode, only_bert=only_bert, cnet_mode=cnet_mode
        )
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
                weight=weight,
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
                use_crf=use_crf,
                train_pairs=(
                    TAXONOMIC_RELS
                    if split_relations == "both"
                    else RELATIONS
                    if split_relations == "pair"
                    else None
                ),
                train_seqs=(
                    CONTEXT_RELS
                    if split_relations == "both"
                    else RELATIONS
                    if split_relations == "seq"
                    else None
                ),
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

    def _ensure_conceptnet(cnet_mode, checkpoint):
        try:
            conceptnet = checkpoint["cnet"]
            if cnet_mode != conceptnet:
                raise ValueError(
                    "The model was not trained using the same configuration for ConceptNet."
                )
        except KeyError:
            if cnet_mode is not None:
                raise ValueError("The model was not trained using ConceptNet.")

    def _run_task(
        tag,
        run_name="ehealth20-default",
        *,
        bert_mode,
        cnet_mode,
        task=None,
        only_bert=False,
    ):
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
            taskB_pair_model = None
            taskB_seq_model = None
        else:
            try:
                checkpoint = torch.load("./trained/taskB.pt")
                _log_checkpoint(checkpoint, desc="Relations (Pairs)")
                _ensure_bert(bert_mode, checkpoint)
                _ensure_conceptnet(cnet_mode, checkpoint)
                taskB_pair_model = checkpoint["model"]
                if taskB_pair_model is not None:
                    taskB_pair_model.eval()
            except FileNotFoundError:
                taskB_pair_model = None

            try:
                checkpoint = torch.load("./trained/taskB-seqs.pt")
                _log_checkpoint(checkpoint, desc="Relations (Sequence)")
                _ensure_bert(bert_mode, checkpoint)
                _ensure_conceptnet(cnet_mode, checkpoint)
                taskB_seq_model = checkpoint["model"]
                if taskB_seq_model is not None:
                    taskB_seq_model.eval()
            except FileNotFoundError:
                taskB_seq_model = None

        algorithm = eHealth20Model(
            taskA_models,
            taskB_pair_model,
            taskB_seq_model,
            bert_mode=bert_mode,
            only_bert=only_bert,
            cnet_mode=cnet_mode,
            tag=tag,
        )

        tasks = handle_args()
        Run.submit(run_name, tasks, algorithm)

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
