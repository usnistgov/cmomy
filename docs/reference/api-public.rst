Top level API (:mod:`cmomy`)
============================

.. automodule:: cmomy
   :no-members:
   :no-inherited-members:
   :no-special-members:




Central Moments wrapper classes
-------------------------------

.. autosummary::
   :toctree: generated/

   CentralMomentsArray
   CentralMomentsData



Legacy wrapper classes
----------------------

.. autosummary::
   :toctree: generated/

   CentralMoments
   xCentralMoments



Factory methods to create wrapper objects
-----------------------------------------

.. autosummary::
   :toctree: generated/

    wrap
    wrap_reduce_vals
    wrap_resample_vals
    wrap_raw
    zeros_like


Reduction routines available at top level (:mod:`.reduction`)
-------------------------------------------------------------

.. autosummary::

   ~reduction.reduce_data
   ~reduction.reduce_data_grouped
   ~reduction.reduce_data_indexed

Resampling routines available at top level (:mod:`.resample`)
-------------------------------------------------------------

.. autosummary::

   ~resample.resample_data
   ~resample.resample_vals
   ~resample.factory_sampler
   ~resample.random_freq
   ~resample.random_indices


Default random number generator (:mod:`cmomy.random`)
-----------------------------------------------------

.. autosummary::

   ~random.default_rng


Concatenation
-------------

.. autosummary::

   ~convert.concat

Convert
-------

.. autosummary::

   convert.moments_type
   convert.cumulative



Confidence intervals (:mod:`.confidence_interval`)
--------------------------------------------------

.. autosummary::

   ~confidence_interval.bootstrap_confidence_interval


Central moment array utilities (:mod:`.utils`)
----------------------------------------------

.. autosummary::

   ~utils.moveaxis
   ~utils.select_moment
   ~utils.assign_moment
   ~utils.vals_to_data



Pre-loaded modules
------------------

.. autosummary::

   ~random
   ~reduction
   ~resample
   ~convert
   ~utils
   ~rolling
