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

   CentralMoments
   xCentralMoments
   CentralMomentsArray
   CentralMomentsData


Concatenation
-------------

.. autosummary::

   ~convert.concat



Reduction routines available at top level (from :mod:`.reduction`)
-----------------------------------------------------------------------

.. autosummary::

   ~reduction.reduce_data
   ~reduction.reduce_data_grouped
   ~reduction.reduce_data_indexed

Resampling routines available at top level (from :mod:`.resample`)
-----------------------------------------------------------------------

.. autosummary::

   ~resample.resample_data
   ~resample.resample_vals
   ~resample.random_freq
   ~resample.random_indices
   ~resample.randsamp_freq
   ~resample.indices_to_freq


Confidence intervals (from :mod:`.confidence_interval`)
------------------------------------------------------------

.. autosummary::

   ~confidence_interval.bootstrap_confidence_interval


Central moment array utilities (from :mod:`.utils`)
--------------------------------------------------------

.. autosummary::

   ~utils.moveaxis
   ~utils.select_moment
   ~utils.assign_moment



Pre-loaded modules
------------------

.. autosummary::

   ~random
   ~reduction
   ~resample
   ~convert
   ~utils
   ~rolling
