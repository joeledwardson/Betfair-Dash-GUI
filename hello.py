from myutils.timing import decorator_timer
import numpy as np

l = range(100000)
npl = np.array(l)


@decorator_timer
def nptestin(x):
    w = np.where(npl == x)[0]
    if w.shape[0]:
        return w[0]
    else:
        print('not found')


@decorator_timer
def testin(x):
    if x in l:
        return l.index(x)
    else:
        print('not found')

print(testin(33333))
print(nptestin(33334))