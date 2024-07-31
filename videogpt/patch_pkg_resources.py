import pkgutil

if not hasattr(pkgutil, 'ImpImporter'):
    class ImpImporter:
        pass

    pkgutil.ImpImporter = ImpImporter

import pkg_resources

# Patch to handle 'FileFinder' issue
import importlib.machinery
importlib.machinery.FileFinder.find_module = importlib.machinery.FileFinder.find_spec

print("Patched pkg_resources successfully.")
