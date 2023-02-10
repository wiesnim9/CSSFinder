from __future__ import annotations

import math
import sys


def invert(c, a):
    return 1 / (c - a)


def listshift(l1, a1):
    return list(map(lambda x: x - a1, l1))


def cov(l1, l2):
    return mean(
        list(
            map(
                lambda x1, x2: x1 * x2, listshift(l1, mean(l1)), listshift(l2, mean(l2))
            )
        )
    )


def trend(l1, l2):
    l1a = list(map(math.log, l1))
    l2a = list(map(math.log, l2))
    return cov(l1a, l2a) / cov(l1a, l1a)


def offset(l1, l2):
    l1a = list(map(math.log, l1))
    l2a = list(map(math.log, l2))
    return mean(l2a) - mean(l1a) * trend(l1, l2)


def findmaximum(ll):
    list1 = list(map(lambda i: i.value, ll))
    list2 = list1[len(list1) // 2 :]

    aaa1 = list2[-1] - 0.000001
    step1 = aaa1 / 10000

    while R(list2, aaa1 - step1) > R(list2, aaa1) and aaa1 > 0:
        aaa1 = aaa1 - step1

    return aaa1


def R(l: list, a: float) -> float:
    ll1 = list(map(lambda x1: invert(x1, a), l))
    return (
        mean(list(map(lambda x1, x2: x1 * x2, ll1, list(range(len(l))))))
        - mean(ll1) * mean(list(range(len(l))))
    ) / math.sqrt(
        (mean(list(map(lambda x: x**2, ll1))) - mean(ll1) ** 2)
        * (
            mean(list(map(lambda x: x**2, list(range(len(l))))))
            - mean(list(range(len(l)))) ** 2
        )
    )


def mean(l):
    return sum(l) / len(l)


def makeshortreport(ll):
    ll10 = list(map(lambda i: i.iteration_number, ll))
    ll11 = list(map(lambda i: i.correction_number, ll))
    kk = findmaximum(ll)
    ll12 = []
    for j1 in range(int(2 * len(ll) / 3), len(ll)):
        ll12.append(ll[j1][2])

    sys.stdout.write("Basing on decay, the squared HS distance is estimsated to be ")
    sys.stdout.write(str(kk))
    sys.stdout.write(" (R=")
    sys.stdout.write(str(R(ll12, kk)))
    sys.stdout.write(")\n")
    sys.stdout.write("The dependence between corrs and trail is approximately:\n")
    sys.stdout.write("corr=trail^")
    sys.stdout.write(str(trend(ll10, ll11)))
    sys.stdout.write("*")
    sys.stdout.write(str(math.exp(offset(ll10, ll11))))
    sys.stdout.write("\n-----------------\n")
