import sys
from pathlib import Path

# The root of the package is two levels above this file: src/
package_root = Path(__file__).resolve().parents[2]
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

try:
    from directed_fuzzing.bp import BP
    from bp.predictors import ConditionalTAGE
    from directed_fuzzing.tage_predictor import TageBP
except ImportError as e:
    print("Failed to import directed_fuzzing modules:", e)
    raise

