# Contract-executor package.
#
# Importable two ways, by design:
#   * As a package  -> `aarch64.contract_executor.saturating_bp` (tests, tools, dev).
#   * As top-level modules from this directory on sys.path -> `import bootstrap_director`,
#     `import saturating_bp` (how the CE binary loads them at runtime via tage_py.c, which
#     adds the executable's own directory to sys.path). This keeps the runtime import
#     location-independent so the files can be copied into a remote-fuzzing bundle anywhere.
#
# To support both, the Python modules here use absolute/self-contained imports (no relative
# imports) — a relative import would break the top-level load path.
