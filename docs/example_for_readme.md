<!-- --- -->
<!-- jupytext: -->
<!--   text_representation: -->
<!--     format_name: myst -->
<!-- kernelspec: -->
<!--   display_name: Python 3 -->
<!--   name: python3 -->
<!-- --- -->


# Example usage


```{eval-rst}
.. ipython:: python
    :suppress:

    import numpy as np

    np.set_printoptions(precision=4)
```

```{eval-rst}
.. ipython:: python

    import numpy as np
    import cmomy

    x = np.random.rand(100)
    m = x.mean()
    mom = np.array([((x - m) ** i).mean() for i in range(4)])
    c = cmomy.CentralMoments.from_vals(x, mom=3)

    mom

    c.cmom()

    # break up into chunks
    c = cmomy.CentralMoments.from_vals(x.reshape(-1, 2), mom=3)
    c
    c.reduce(axis=0).cmom()

    # unequal chunks
    x0, x1, x2 = x[:20], x[20:60], x[60:]

    cs = [cmomy.CentralMoments.from_vals(_, mom=3) for _ in (x0, x1, x2)]

    c = cs[0] + cs[1] + cs[2]

    c.cmom()

```

<!-- ```{code-cell} ipython3 -->
<!-- b = 2 -->
<!-- print('b value', b) -->
<!-- ``` -->

<!-- ```python -->
<!-- import numpy as np -->

<!-- for x in [1, 2, 3]: -->
<!--     pass -->
<!-- ``` -->
