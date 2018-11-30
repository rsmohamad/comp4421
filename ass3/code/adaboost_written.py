import numpy as np
import math

y_train = np.array([1, 1, -1, -1, 1, 1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, 1, -1, 1, -1, 1])
h2 = np.array([-1, 1, -1, 1, -1, -1, 1, -1, -1])
h3 = np.array([1, 1, -1, 1, -1, -1, -1, 1, -1])
h4 = np.array([-1, -1, 1, -1, 1, -1, 1, -1, 1])
h5 = np.array([-1, 1, 1, -1, 1, -1, 1, -1, 1])
responses = [h1, h2, h3, h4, h5]
sample_weights = np.ones(y_train.shape[0])
clf_W = []

for i in range(len(responses)):
    sample_weights = sample_weights/np.sum(sample_weights)

    print("T = %d" % (i+1))
    print("Sample weights:")
    print(sample_weights)
    response = responses[i]
    not_eq = np.not_equal(y_train, response)
    weighted = np.multiply(sample_weights, not_eq)
    err = np.sum(weighted)
    a = 0.5 * math.log((1-err)/err)
    clf_W.append(a)

    print("Error = %f" % (err))
    print("Classifier weight = %f" % (a))

    weight_update = np.multiply(not_eq, a)
    weight_update = np.exp(weight_update)
    sample_weights = np.multiply(sample_weights, weight_update)

    print()

pos_idx = np.nonzero(np.array(clf_W) > 0)[0]
res = np.zeros(y_train.shape[0])
for i in pos_idx:
    i = int(i)
    res = res + clf_W[i] * responses[i]

res = np.sign(res).astype(int)
print(res)

