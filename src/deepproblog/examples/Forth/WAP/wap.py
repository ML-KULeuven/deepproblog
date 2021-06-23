from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Forth.WAP.wap_network import get_networks
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

train_queries = QueryDataset("data/train.pl")
dev_queries = QueryDataset("data/dev.pl")
test_queries = QueryDataset("data/test.pl")

networks = get_networks(0.005, 0.5)

train_networks = [Network(x[0], x[1], x[2]) for x in networks]
test_networks = [Network(networks[0][0], networks[0][1])] + [
    Network(x[0], x[1], k=1) for x in networks[1:]
]

model = Model("wap.pl", train_networks)
model.set_engine(ExactEngine(model), cache=True)

test_model = Model("wap.pl", test_networks)
test_model.set_engine(ExactEngine(test_model), cache=False)

train_obj = train_model(
    model,
    DataLoader(train_queries, 10),
    40,
    log_iter=10,
    test=lambda x: [
        ("Accuracy", get_confusion_matrix(test_model, test_queries).accuracy())
    ],
    test_iter=30,
)
