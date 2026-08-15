"""Generic, self-contained document logic for the Mudqeq AI web demo.

Nothing in this package imports from the desktop application. The generic
extraction/chunking logic was reimplemented here (not imported) so the Docker
build context can be scoped strictly to ``web_demo/``.
"""
