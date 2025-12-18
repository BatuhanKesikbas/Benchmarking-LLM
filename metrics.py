# metrics.py
from typing import Tuple, List, Dict, Any
#pred=tahmin edilen cevap, gold=olması gereken cevap.
def compute_accuracy(pred_letter: str, gold_letter: str) -> int:
    """
    Compare predicted single letter with gold letter.
    Returns 1 if equal (case-insensitive), else 0.
    If prediction is empty, returns 0.
    """
    if not pred_letter or not gold_letter:
        return 0
    return 1 if pred_letter.strip().upper() == gold_letter.strip().upper() else 0

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate results per model: compute accuracy (mean of is_correct).
    results: list of dicts containing at least keys: 'model' and 'is_correct' (0/1)
    Returns dict mapping model -> accuracy (0..100 percent)
    """
    from collections import defaultdict
    counts = defaultdict(int)
    corrects = defaultdict(int)
    for r in results:
        m = r.get("model", "unknown")
        counts[m] += 1
        corrects[m] += int(r.get("is_correct", 0))
    agg = {}
    for m in counts:
        agg[m] = (corrects[m] / counts[m]) * 100.0 if counts[m] > 0 else 0.0
    return agg
