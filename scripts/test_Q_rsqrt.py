import random
from struct import pack, unpack
import timeit
from math import inf
from alive_progress import alive_it


def bar(iterator, title=None):
    """ wrap the iterator as a progress-bar """
    return alive_it(iterator, title=title, bar='bubbles', spinner='horizontal')

def Q_rsqrt(number):
    i = unpack('=l', pack('=f', number))[0]
    i = 0x5f3759df - (i >> 1)
    y = unpack('=f', pack('=l', i))[0]
    y = y * (1.5 - (0.5 * number) * y * y)
    # y = y * (1.5 - (0.5 * number) * y * y)
    return y


def test(numbers):
    for number in numbers:
        _ = Q_rsqrt(number)
        # yield _


def test2(numbers):
    for number in numbers:
        _ = 1 / number ** 0.5
        # yield _



numbers = [random.uniform(-1e20, 1e20) for _ in range(int(1e8))]
# # print(numbers)
# for i, j in bar(zip(test(numbers), test2(numbers)), 'testing !'):
#     if abs(i - j) > 1e-20:
#         print(i, j, i - j)

starttime = timeit.default_timer()
test(numbers)
print(f'{timeit.default_timer() - starttime} s')
starttime = timeit.default_timer()
test2(numbers)
print(f'{timeit.default_timer() - starttime} s')
