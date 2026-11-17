"""DreaMS (Deep Representations Empowering the Annotation of Mass Spectra).

This package registers a lightweight compatibility shim so that code written
against older versions of ``matchms`` (which imported
``matchms.similarity.ModifiedCosine``) continues to work with newer ``matchms``
releases that renamed that class to ``ModifiedCosineGreedy``.

This is required both for this package and for the external ``msml``
dependency, which is imported when unpickling the published DreaMS checkpoints
(saved with older torch/matchms).
"""
try:
    from matchms.similarity import ModifiedCosineGreedy as _ModifiedCosineGreedy
    import matchms.similarity as _matchms_similarity
    if not hasattr(_matchms_similarity, "ModifiedCosine"):
        _matchms_similarity.ModifiedCosine = _ModifiedCosineGreedy
except Exception:  # pragma: no cover - best-effort shim
    pass