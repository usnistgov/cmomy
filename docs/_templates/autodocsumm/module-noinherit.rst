

.. automodule:: {{ fullname }}
   :autosummary:
   :members:


{% block modules %}
{% if modules %}
.. autosummary::
   :toctree:
   :template: autodocsumm/module-noinherit.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
