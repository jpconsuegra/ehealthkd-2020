import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from kdtools.datasets import LabeledSentencesDS, to_biluov
from kdtools.models import BasicSequenceTagger
from scripts.submit import Algorithm
from scripts.utils import Collection


class UHMajaModel(Algorithm):
    CHAR_EMBEDDING_DIM = 100
    CHAR_REPR_DIM = 200
    TOKEN_REPR_DIM = 300

    def __init__(self):
        self.model_taskA = None

    def run(self, collection: Collection, *args, taskA: bool, taskB: bool, **kargs):
        return super().run(collection, *args, taskA=taskA, taskB=taskB, **kargs)

    def train(self, collection: Collection, n_epochs=100):
        self.model = self.train_taskA(collection, n_epochs)

    def train_taskA(self, collection: Collection, n_epochs=100):
        sentences = [s.text for s in collection.sentences]
        entities = [[k.spans for k in s.keyphrases] for s in collection.sentences]
        dataset = LabeledSentencesDS(sentences, lambda ss: to_biluov(ss, entities))

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
                dataset.shallow_dataloader(shuffle=True), total=len(dataset)
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
