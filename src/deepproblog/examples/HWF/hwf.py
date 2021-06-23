from json import dumps
from sys import argv

from torch.optim import Adam

from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.dataset import DataLoader
from deepproblog.train import train_model
from deepproblog.examples.HWF.data import HWFDataset, hwf_images
from deepproblog.examples.HWF.network import SymbolEncoder, SymbolClassifier
from deepproblog.heuristics import *
from deepproblog.utils import format_time_precise, get_configuration, config_to_string

i = int(argv[1]) if len(argv) > 1 else 0
configurations = {
    "method": ["exact"],
    "curriculum": [False],
    "N": [1, 3],
    "run": range(5),
}
configuration = get_configuration(configurations, i)

name = "hwf_" + config_to_string(configuration) + "_" + format_time_precise()
torch.manual_seed(configuration["run"])

N = configuration["N"]

if configuration["method"] == "exact":
    if N > 3:
        exit()

curriculum = configuration["curriculum"]
print("Training HWF with N={} and curriculum={}".format(N, curriculum))

encoder = SymbolEncoder()
network1 = SymbolClassifier(encoder, 10)
network2 = SymbolClassifier(encoder, 4)

net1 = Network(network1, "net1", Adam(network1.parameters(), lr=3e-3), batching=True)
net2 = Network(network2, "net2", Adam(network2.parameters(), lr=3e-3), batching=True)

model = Model("model.pl", [net1, net2])
model.add_tensor_source("hwf", hwf_images)
heuristic = GeometricMean()
if configuration["method"] == "exact":
    model.set_engine(ExactEngine(model), cache=True)
elif configuration["method"] == "approximate":
    model.set_engine(
        ApproximateEngine(
            model, 1, heuristic, timeout=30, ignore_timeout=True, exploration=True
        )
    )

if curriculum:
    dataset = HWFDataset("train2", lambda x: x <= N)
    val_dataset = HWFDataset("val", lambda x: x <= N)
    test_dataset = HWFDataset("test", lambda x: x <= N)
else:
    dataset = HWFDataset("train2", lambda x: x == N)
    val_dataset = HWFDataset("val", lambda x: x == N)
    test_dataset = HWFDataset("test", lambda x: x == N)
loader = DataLoader(dataset, 32, shuffle=True)
print("Training on size {}".format(N))
train_log = train_model(
    model,
    loader,
    50,
    log_iter=50,
    inital_test=False,
    test_iter=100,
    test=lambda x: [
        ("Val_accuracy", get_confusion_matrix(x, val_dataset, eps=1e-6).accuracy()),
        ("Test_accuracy", get_confusion_matrix(x, test_dataset, eps=1e-6).accuracy()),
    ],
)

model.save_state("models/" + name + ".pth")
final_acc = get_confusion_matrix(model, test_dataset, eps=1e-6, verbose=0).accuracy()
train_log.logger.comment("Accuracy {}".format(final_acc))
train_log.logger.comment(dumps(model.get_hyperparameters()))
train_log.write_to_file("log/" + name)
