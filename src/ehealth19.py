import torch
import torch.optim as optim
from kdtools.datasets import BILUOVSentencesDS, from_biluov
from kdtools.models import BasicSequenceTagger
from kdtools.utils import train_on_shallow_dataloader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from scripts.submit import Algorithm, Run, handle_args
from scripts.utils import ENTITIES, RELATIONS, Collection, Keyphrase, Relation


class UHMajaModel(Algorithm):
    CHAR_EMBEDDING_DIM = 100
    CHAR_REPR_DIM = 200
    TOKEN_REPR_DIM = 300

    def __init__(self):
        self.taskA_models = {}

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
        dataset = BILUOVSentencesDS([s.text for s in collection.sentences])

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

    def train(self, collection: Collection, n_epochs=100):
        self.train_taskA(collection, n_epochs)

    def train_taskA(self, collection: Collection, n_epochs=100):
        for label in ENTITIES:
            self.taskA_models[label] = self.build_taskA_model(
                collection, label, n_epochs
            )

    def build_taskA_model(
        self, collection: Collection, entity_label: str, n_epochs=100
    ):
        sentences = [s.text for s in collection.sentences]
        entities = [
            [k.spans for k in s.keyphrases if k.label == entity_label]
            for s in collection.sentences
        ]
        dataset = BILUOVSentencesDS(sentences, entities)

        model = BasicSequenceTagger(
            char_vocab_size=dataset.char_size,
            char_embedding_dim=self.CHAR_EMBEDDING_DIM,
            padding_idx=dataset.padding,
            char_repr_dim=self.CHAR_REPR_DIM,
            word_repr_dim=dataset.vectors_len,
            postag_repr_dim=dataset.pos_size,
            token_repr_dim=self.TOKEN_REPR_DIM,
            num_labels=dataset.label_size,
        )

        train_on_shallow_dataloader(
            model,
            dataset,
            optim=optim.SGD,
            criterion=CrossEntropyLoss,
            n_epochs=n_epochs,
            desc=entity_label,
        )

        return model


if __name__ == "__main__":
    from pathlib import Path

    algorithm = UHMajaModel()

    training = Collection().load(Path("data/training/scenario.txt"))
    algorithm.train(training)

    tasks = handle_args()
    Run.submit("ehealth19-maja", tasks, algorithm)
