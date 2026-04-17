"""NFS (Native Fold Score) Pipeline — CoT folding analysis framework.

Modules:
    graph_builder     Phase 1: slice-level graph construction
    primitives        Phase 2: structural primitive extraction
    fold_score        Phase 3: NFS scoring + validation
    segment_graph     Phase 4: segment-level graph construction
    segment_primitives Phase 5: segment-level primitive extraction
    segment_score     Phase 6: segment-level NFS scoring
"""

import sys
from pathlib import Path

# Ensure repo root is in sys.path so that project_paths, alignment, hmm_simple
# are importable when this package is used from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
