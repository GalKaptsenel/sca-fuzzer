import sys
from pathlib import Path

print("directed_fuzzing init")
# The root of the package is two levels above this file: src/
package_root = Path(__file__).resolve().parents[2]
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

try:
    from directed_fuzzing.bp import BP
    from directed_fuzzing.saturating_bp import Aarch64NeoverseN3BPU, SaturatingCounterBPCommon
except ImportError as e:
    print("Failed to import directed_fuzzing modules:", e)
    raise

