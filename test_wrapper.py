
import datetime
from multiprocessing import Pool

import numpy as np

from basic_test import *


def mp_wrapper(learning_rate):
	train_and_test_agent(learning_rate=learning_rate)


learning_rates = np.power(10, -(2 + np.random.random_sample(size=20) * 2))


p = Pool(20)  # number of cores - 1

t1 = datetime.datetime.now()
p.map(mp_wrapper, learning_rates)
t2 = datetime.datetime.now() - t1

print('Test time:')
print(t2)
