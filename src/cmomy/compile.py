# type: ignore  # noqa: PGH003
"""Set of routines to call most cases to pre-compile numba functions."""

from __future__ import annotations


def compile_numba_funcs(verbose: bool = True) -> None:  # noqa: ARG001
    """Compile commonly used numba functions."""

    # def _print(*args: Any, **kwargs: Any) -> None:
    #     if verbose:
    #         print(*args, **kwargs)
    #     else:
    #         pass

    # def _get_val(shape, dtype=None) -> Iterator[np.ndarray]:
    #     val = np.zeros(val_shape, dtype=dtype)

    #     for i in range(2, len(shape)):
    #         yield np.moveaxis(val, i, 0)

    # def _do_val(s: CentralMoments, val, w, data, v=None):

    #     s.push_val(val)
    #     s.push_val(val, w=w)
    #     s.push_data(data)

    #     if v:
    #         s.push_stat(val, v=v, w=w)

    #     rmom = s.rmom()
    #     for i in range(s.val_ndim):
    #         _ = CentralMoments.from_raw(np.moveaxis(rmom, i, 0), mom=s.mom)

    # def _do_vals(s: CentralMoments, vals, ws, datas, vs=None):
    #     s.push_vals(vals)
    #     s.push_vals(vals, w=w)
    #     s.push_datas(datas)

    #     if vs:
    #         s.push_stats(vals, vs)

    # dtype = float

    # vals_shapes = [(), (4,), (4, 4), (4, 4, 4)]

    # moms = [(4,), (4,4), (1,), (1, 1)]

    # I = list(product(vals_shapes, moms))

    # for i, (val_shape, mom) in enumerate(I):
    #     s = CentralMoments.zeros(val_shape=val_shape, mom=mom)

    #     for val in _get_val(s.val_shape):
    #         s.push_val(val)
    #         s.push_stat(val)
    #         for w in _get_val(s.val_shape):
    #             s.push_val(val, w=w)

    # orders = ["C", "F"]

    # vmo = list(product(vals_shapes, moms, orders, orders, orders))

    # total = len(vmo)

    # tic = time.perf_counter()

    # numba_random_seed(0)

    # for i, (val_shape, mom, worder, vorder, vorder2) in enumerate(vmo):
    #     _print(f"Run {i}/{total} ...", end=" ")

    #     # "val"
    #     s = CentralMoments.zeros(val_shape=val_shape, mom=mom)

    #     val = np.random.rand(*val_shape)
    #     w = np.random.rand(*val_shape)
    #     data = np.random.rand(*s.shape)

    #     val = np.asarray(val, order=vorder, dtype=dtype)
    #     w = np.asarray(w, order=worder, dtype=dtype)
    #     data = np.asarray(data, order=vorder, dtype=dtype)

    #     if len(mom) == 2:
    #         val2 = np.random.rand(*val_shape)
    #         val2 = np.asarray(val, order=vorder2, dtype=dtype)

    #         val = (val, val2)

    #     s.push_val(val)
    #     for a in range(len(vals_shape)):
    #         vv = np.moveaxis(val, 0, )
    #         s.push_val(val, w=w)
    #     s.push_data(data)

    #     # convert
    #     rmom = s.rmom()
    #     _ = CentralMoments.from_raw(np.asarray(rmom, order=vorder), mom=mom)

    #     if len(mom) == 1:
    #         s.push_stat(val)

    #     # "vals"
    #     for axis in [0, -1]:
    #         if axis == 0:
    #             vals = np.random.rand(*((4,) + val_shape))
    #             w = np.random.rand(*vals.shape)
    #             datas = np.random.rand(*((4,) + s.shape))
    #         elif axis == -1:
    #             vals = np.random.rand(*(val_shape + (4,)))
    #             w = np.random.rand(*vals.shape)
    #             datas = np.random.rand(*(s.val_shape + (4,) + s.mom_shape))

    #         vals = np.asarray(vals, order=vorder)
    #         w = np.asarray(w, order=worder)
    #         datas = np.asarray(datas, order=vorder)

    #         if len(mom) == 2:
    #             vals2 = np.random.rand(*vals.shape)
    #             vals2 = np.asarray(vals, order=vorder2, dtype=dtype)
    #             vals = (vals, vals2)

    #         s.push_vals(vals, axis=axis)
    #         s.push_vals(vals, w=w, axis=axis)
    #         s.push_datas(datas, axis=axis)

    #         if len(mom) == 1:
    #             s.push_stats(vals, axis=axis)

    #     # resamplin
    #     for axis_resamp in range(len(val_shape)):
    #         for parallel in [False]:
    #             s.resample_and_reduce(nrep=2, parallel=parallel, axis=axis_resamp)

    #             # resample vals
    #             s = CentralMoments.from_resample_vals(
    #                 vals, nrep=2, mom=mom, axis=axis_resamp, parallel=parallel
    #             )
    #     print("done")

    # toc = time.perf_counter()

    # _print(f"Finished {total} commands in {toc-tic:0.4f} seconds")


if __name__ == "__main__":
    compile_numba_funcs()
