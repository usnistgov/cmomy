# mypy: disable-error-code="no-untyped-def"
"""Numba functions to convert between raw and central moments."""

from .utils import BINOMIAL_FACTOR, myjit


@myjit()
def central_to_raw_moments(central, raw) -> None:
    nv = central.shape[0]
    order = central.shape[1] - 1

    for v in range(nv):
        ave = central[v, 1]

        raw[v, 0] = central[v, 0]
        raw[v, 1] = ave

        for n in range(2, order + 1):
            tmp = 0.0
            ave_i = 1.0
            for i in range(n - 1):
                tmp += central[v, n - i] * ave_i * BINOMIAL_FACTOR[n, i]
                ave_i *= ave

            # last two
            # <dx> = 0 so skip i = n-1
            # i = n
            tmp += ave_i * ave
            raw[v, n] = tmp


@myjit()
def raw_to_central_moments(raw, central) -> None:
    nv = central.shape[0]
    order = central.shape[1] - 1

    for v in range(nv):
        ave = raw[v, 1]

        central[v, 0] = raw[v, 0]
        central[v, 1] = ave

        for n in range(2, order + 1):
            tmp = 0.0
            ave_i = 1.0
            for i in range(n - 1):
                tmp += raw[v, n - i] * ave_i * BINOMIAL_FACTOR[n, i]
                ave_i *= -ave

            # last two
            # right now, ave_i = (-ave)**(n-1)
            # i = n-1
            # ave * ave_i * n
            # i = n
            # 1 * (-ave) * ave_i
            tmp += ave * ave_i * (n - 1)
            central[v, n] = tmp


# comoments
@myjit()
def central_to_raw_comoments(central, raw) -> None:
    nv = central.shape[0]
    order0 = central.shape[1] - 1
    order1 = central.shape[2] - 1

    for v in range(nv):
        ave0 = central[v, 1, 0]
        ave1 = central[v, 0, 1]

        for n in range(order0 + 1):
            for m in range(order1 + 1):
                nm = n + m
                if nm <= 1:
                    raw[v, n, m] = central[v, n, m]
                else:
                    tmp = 0.0
                    ave_i = 1.0
                    for i in range(n + 1):
                        ave_j = 1.0
                        for j in range(m + 1):
                            nm_ij = nm - (i + j)
                            if nm_ij == 0:
                                # both zero order
                                tmp += ave_i * ave_j
                            elif nm_ij == 1:
                                # <dx**0 * dy**1> = 0
                                pass
                            else:
                                tmp += (
                                    central[v, n - i, m - j]
                                    * ave_i
                                    * ave_j
                                    * BINOMIAL_FACTOR[n, i]
                                    * BINOMIAL_FACTOR[m, j]
                                )
                            ave_j *= ave1
                        ave_i *= ave0
                    raw[v, n, m] = tmp


@myjit()
def raw_to_central_comoments(raw, central) -> None:
    nv = central.shape[0]
    order0 = central.shape[1] - 1
    order1 = central.shape[2] - 1

    for v in range(nv):
        ave0 = raw[v, 1, 0]
        ave1 = raw[v, 0, 1]

        for n in range(order0 + 1):
            for m in range(order1 + 1):
                nm = n + m
                if nm <= 1:
                    central[v, n, m] = raw[v, n, m]
                else:
                    tmp = 0.0
                    ave_i = 1.0
                    for i in range(n + 1):
                        ave_j = 1.0
                        for j in range(m + 1):
                            nm_ij = nm - (i + j)
                            if nm_ij == 0:
                                # both zero order
                                tmp += ave_i * ave_j
                            else:
                                tmp += (
                                    raw[v, n - i, m - j]
                                    * ave_i
                                    * ave_j
                                    * BINOMIAL_FACTOR[n, i]
                                    * BINOMIAL_FACTOR[m, j]
                                )
                            ave_j *= -ave1
                        ave_i *= -ave0
                    central[v, n, m] = tmp
