import torch
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
from kdtools.models import BasicSequenceTagger, BasicSequenceClassifier
from kdtools.utils import (
    jointly_train_on_shallow_dataloader,
    train_on_shallow_dataloader,
)
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from scripts.submit import Algorithm, Run, handle_args
from scripts.utils import ENTITIES, RELATIONS, Collection, Keyphrase, Relation


class UHMajaModel(Algorithm):
    CHAR_EMBEDDING_DIM = 100
    CHAR_REPR_DIM = 200
    TOKEN_REPR_DIM = 300

    def __init__(self):
        self.nlp = get_nlp()
        self.taskA_models: dict = None
        self.taskB_model = None

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

    def train(self, collection: Collection, *, jointly, inclusion, n_epochs=100):
        self.train_taskA(collection, jointly, n_epochs)
        self.train_taskB(collection, jointly, inclusion, n_epochs)

    def train_taskA(self, collection: Collection, jointly, n_epochs=100):
        char_encoder = None

        models = {}
        datasets = {}
        for label in ENTITIES:
            model, dataset = self.build_taskA_model(
                collection, label, n_epochs, shared=char_encoder
            )

            models[label] = model
            datasets[label] = dataset

            char_encoder = model.char_encoder if jointly else None

        if jointly:
            # dicts are stable
            self.train_all_taskA_models(
                models.values(), datasets.values(), "all", n_epochs
            )
        else:
            for label in ENTITIES:
                self.train_taskA_model(models[label], datasets[label], label, n_epochs)

        self.taskA_models = models

    def build_taskA_model(
        self, collection: Collection, entity_label: str, n_epochs=100, shared=None
    ):
        sentences = [s.text for s in collection.sentences]
        entities = [
            [k.spans for k in s.keyphrases if k.label == entity_label]
            for s in collection.sentences
        ]
        dataset = BILUOVSentencesDS(sentences, entities, language=self.nlp)

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

        return model, dataset

    def train_taskA_model(self, model, dataset, desc, n_epochs=100):
        train_on_shallow_dataloader(
            model,
            dataset,
            optim=optim.SGD,
            criterion=CrossEntropyLoss,
            n_epochs=n_epochs,
            desc=desc,
        )

    def train_all_taskA_models(self, models, datasets, desc, n_epochs=100):
        jointly_train_on_shallow_dataloader(
            models,
            datasets,
            optim=optim.SGD,
            criterion=CrossEntropyLoss,
            n_epochs=n_epochs,
            desc=desc,
        )

    def train_taskB(self, collection: Collection, jointly, inclusion, n_epochs=100):
        nlp = self.nlp

        tokensxsentence = [nlp(s.text) for s in collection.sentences]
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
            nlp,
            inclusion=inclusion,
            char2repr=None,
        )

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

        train_on_shallow_dataloader(model, dataset, n_epochs=n_epochs, desc="relations")

        self.taskB_model = model


if __name__ == "__main__":
    from pathlib import Path

    algorithm = UHMajaModel()

    training = Collection().load(Path("data/training/scenario.txt"))
    algorithm.train(training, jointly=True, inclusion=0.1)

    tasks = handle_args()
    Run.submit("ehealth19-maja", tasks, algorithm)
