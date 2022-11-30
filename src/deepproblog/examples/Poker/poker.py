import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Poker import PokerSeparate
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.optimizer import SGD
from deepproblog.train import train_model
from deepproblog.utils.standard_networks import smallnet

batch_size = 50
datasets = {
    "unfair": PokerSeparate(
        "unfair", probs=[0.2, 0.4, 0.15, 0.25], extra_supervision=True
    ),
    "fair_test": PokerSeparate("fair_test"),
}

dataset = "unfair"
net = Network(
    smallnet(pretrained=True, num_classes=4, size=(100, 150)), "net1", batching=True
)
net.optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
loader = DataLoader(datasets[dataset], batch_size)

model = Model("model.pl", [net])
model.set_engine(ExactEngine(model), cache=True)
model.optimizer = SGD(model, 5e-2)
model.add_tensor_source(dataset, datasets[dataset])
model.add_tensor_source("fair_test", datasets["fair_test"])

train_obj = train_model(
    model,
    loader,
    10,
    loss_function_name="mse",
    log_iter=len(datasets["unfair"]) // batch_size,
    test_iter=5 * len(datasets["unfair"]) // batch_size,
    test=lambda x: [
        ("Accuracy", get_confusion_matrix(model, datasets["fair_test"]).accuracy())
    ],
    infoloss=0.5,
)  # ,

cm = get_confusion_matrix(model, datasets["fair_test"])
