from collections import defaultdict
from statistics import mean
from typing import Optional

from deepproblog.dataset import Dataset
from deepproblog.model import Model
from deepproblog.utils.confusion_matrix import ConfusionMatrix


def get_confusion_matrix(
    model: Model, dataset: Dataset, verbose: int = 0, eps: Optional[float] = None
) -> ConfusionMatrix:
    """

    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate the model on.
    :param verbose: Set the verbosity. If verbose > 0, then print confusion matrix and accuracy.
    If verbose > 1, then print all wrong answers.
    :param eps: If set, then the answer will be treated as a float, and will be considered correct if
    the difference between the predicted and ground truth value is smaller than eps.
    :return: The confusion matrix when evaluating model on dataset.
    """
    confusion_matrix = ConfusionMatrix()
    model.eval()
    for i, gt_query in enumerate(dataset.to_queries()):
        test_query = gt_query.variable_output()
        answer = model.solve([test_query])[0]
        actual = str(gt_query.output_values()[0])
        if len(answer.result) == 0:
            predicted = "no_answer"
            if verbose > 1:
                print("no answer for query {}".format(gt_query))
        else:
            max_ans = max(answer.result, key=lambda x: answer.result[x])
            p = answer.result[max_ans]
            if eps is None:
                predicted = str(max_ans.args[gt_query.output_ind[0]])
            else:
                predicted = float(max_ans.args[gt_query.output_ind[0]])
                actual = float(gt_query.output_values()[0])
                if abs(actual - predicted) < eps:
                    predicted = actual
            if verbose > 1 and actual != predicted:
                print(
                    "{} {} vs {}::{} for query {}".format(
                        i, actual, p, predicted, test_query
                    )
                )
        confusion_matrix.add_item(predicted, actual)

    if verbose > 0:
        print(confusion_matrix)
        print("Accuracy", confusion_matrix.accuracy())

    return confusion_matrix


def get_fact_accuracy(
    model: Model, dataset: Dataset, verbose: int = 0, threshold: float = 0.5
) -> ConfusionMatrix:
    """

    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate the model on.
    :param verbose: Set the verbosity. If verbose > 0, then print confusion matrix and accuracy.
    If verbose > 1, then print all wrong answers.
    :param threshold: If set, then the answer will be treated as a float, and will be considered correct if
    the difference between the predicted and ground truth value is smaller than eps.
    :return: The confusion matrix when evaluating model on dataset.
    """
    confusion_matrix = ConfusionMatrix()
    model.eval()
    probabilities = defaultdict(list)
    for i, query in enumerate(dataset.to_queries()):
        answer = model.solve([query])[0]
        if len(answer.result) == 0:
            predicted = "no_answer"
            if verbose > 1:
                print("no answer for query {}".format(query))
        else:
            predicted_p = float(answer.result[query.substitute().query])
            predicted = "true" if predicted_p >= threshold else "false"
            probabilities[predicted].append(predicted_p)
        actual = "true" if query.p >= threshold else "false"
        confusion_matrix.add_item(predicted, actual)
    for c in probabilities:
        print("Average probability for class {}: {}".format(c, mean(probabilities[c])))
    if verbose > 0:
        print(confusion_matrix)
        print("Accuracy", confusion_matrix.accuracy())

    return confusion_matrix


# def multi_class_accuracy(model, data, test=False, verbose=1, inds=None):
#     if inds is None:
#         inds = [-1]
#     correct = 0
#     sub = None
#     N = 0
#     model.eval()
#     for batch in data:
#         N += len(batch)
#         query_batch = []
#         for q in batch:
#             if type(q) is tuple:
#                 q, sub = q
#             args = list(q.args)
#             for i in inds:
#                 args[i] = Var('X_{}'.format(i))
#             q = q(*args)
#             if sub is not None:
#                 q = (q, sub)
#             query_batch.append(q)
#
#         result, _ = model.solve(query_batch, test)
#         for i, out in enumerate(result):
#             out = max(out, key=lambda x: out[x][0])
#             if type(batch[i]) is tuple:
#                 res_out = batch[i][0]
#                 if test:
#                     res_out = res_out.apply_term(batch[i][1])
#             else:
#                 res_out = batch[i]
#             if out == res_out:
#                 correct += 1
#                 if verbose > 2:
#                     print('Correct', out)
#             else:
#                 if verbose > 1:
#                     print('Wrong', res_out, 'vs', out)
#     if verbose > 0:
#         print('Accuracy', correct / N)
#     return [('Accuracy', correct / N)]
#
#
# def fact_accuracy(model, data, test=False, threshold=0.5, verbose=1):
#     correct = 0
#     N = 0
#     model.eval_mode()
#     for batch in data:
#         N += len(batch)
#         result, _ = model.solve(batch, test)
#         for i, out in enumerate(result):
#             if type(batch[i]) is tuple:
#                 q = batch[i][0]
#             else:
#                 q = batch[i]
#             p = out[q][0]
#             if p >= threshold:
#                 correct += 1
#             if verbose > 1:
#                 print(p, q)
#     if verbose > 0:
#         print('Accuracy', correct / N)
#     return [('Accuracy', correct / N)]
#
#
