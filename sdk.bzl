"""Repository paths used by native C++ targets."""
SDK_ROOT = Label("@sdk//:sdk").workspace_root
SDK_REPO = Label("@sdk//:sdk").workspace_name
ARROW_REPO = Label("@arrow_cpu//:arrow").workspace_name
