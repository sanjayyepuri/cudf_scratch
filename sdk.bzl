"""Locations of the native SDK repository installed by rules_conda."""
SDK_ROOT = Label("@sdk//:sdk").workspace_root
SDK_REPO = Label("@sdk//:sdk").workspace_name
