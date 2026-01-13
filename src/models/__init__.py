# Import to register models with AutoModel
try:
    from . import gritlm7b
except (ImportError, TypeError):
    pass

try:
    from . import nvmmembed
except (ImportError, TypeError):
    pass
