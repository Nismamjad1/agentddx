from datasets import load_dataset
import numpy as np

ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
print("fields:", ds.column_names)

lens = [len(q["question"]) for q in ds]
a = np.array(lens)

print(f"\nn={len(a)}")
print(f"mean={a.mean():.0f}  median={np.median(a):.0f}  std={a.std():.0f}")
print(f"p10={np.percentile(a,10):.0f}  p25={np.percentile(a,25):.0f}  "
      f"p75={np.percentile(a,75):.0f}  p90={np.percentile(a,90):.0f}")
print(f"min={a.min()}  max={a.max()}")
print(f"<=120 chars: {(a<=120).sum()} ({100*(a<=120).mean():.1f}%)")
print(f"120 chars = {100*120/a.mean():.1f}% of the mean stem")
