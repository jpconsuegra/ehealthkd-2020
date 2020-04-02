import torch
import torch.optim as optim
from kdtools.datasets import BILUOVSentencesDS, from_biluov
from kdtools.models import BasicSequenceTagger
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
        self.model = self.train_taskA(collection, n_epochs)

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
            char_vocab_size=len(dataset.char_vocab),
            char_embedding_dim=self.CHAR_EMBEDDING_DIM,
            padding_idx=dataset.PADDING,
            char_repr_dim=self.CHAR_REPR_DIM,
            word_repr_dim=dataset.vectors_len,
            postag_repr_dim=len(dataset.pos2index),
            token_repr_dim=self.TOKEN_REPR_DIM,
            num_labels=len(dataset.label2index),
        )

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = CrossEntropyLoss()
        for epoch in range(n_epochs):

            correct = 0
            total = 0
            running_loss = 0.0

            for data in tqdm(
                dataset.shallow_dataloader(shuffle=True),
                total=len(dataset),
                desc=entity_label,
            ):
                *sentence, label = data
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model(sentence).squeeze(0)

                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                predicted = torch.argmax(output, -1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            print(f"[{epoch + 1}] loss: {running_loss / len(dataset) :0.3}")
            print(f"[{epoch + 1}] accuracy: {correct / total}")

        return model


if __name__ == "__main__":
    from pathlib import Path

    algorithm = UHMajaModel()

    training = Collection().load(Path("data/training/scenario.txt"))
    algorithm.train(training)

    tasks = handle_args()
    Run.submit("ehealth19-maja", tasks, algorithm)
