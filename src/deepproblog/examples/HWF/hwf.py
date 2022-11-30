from json import dumps
from torch.optim import Adam

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.HWF.data import HWFDataset, hwf_images
from deepproblog.examples.HWF.network import SymbolEncoder, SymbolClassifier
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

N = 1
method = 'exact'
name = "hwf_{}_{}".format(method, N)
curriculum = False

print("Training HWF with N={} and curriculum={}".format(N, curriculum))

encoder = SymbolEncoder()
network1 = SymbolClassifier(encoder, 10)
network2 = SymbolClassifier(encoder, 4)

net1 = Network(network1, "net1", Adam(network1.parameters(), lr=3e-3), batching=True)
net2 = Network(network2, "net2", Adam(network2.parameters(), lr=3e-3), batching=True)

model = Model("model.pl", [net1, net2])
model.add_tensor_source("hwf", hwf_images)

if method == "exact":
    model.set_engine(ExactEngine(model), cache=True)
elif method == "approximate":
    heuristic = ApproximateEngine.geometric_mean
    model.set_engine(ApproximateEngine(model, 1, heuristic, timeout=30, ignore_timeout=True, exploration=True))

try:
    if curriculum:
        dataset = HWFDataset("train2", lambda x: x <= N)
        val_dataset = HWFDataset("val", lambda x: x <= N)
        test_dataset = HWFDataset("test", lambda x: x <= N)
    else:
        dataset = HWFDataset("train2", lambda x: x == N)
        val_dataset = HWFDataset("val", lambda x: x == N)
        test_dataset = HWFDataset("test", lambda x: x == N)
except FileNotFoundError:
    print('The HWD dataset has not been downloaded. See the README.md for info on how to download it.')
    dataset, val_dataset, test_dataset = None, None, None
    exit(1)

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
