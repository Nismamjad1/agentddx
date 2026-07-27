import json
import glob
import os

def analyze_json(filepath):
    filename = os.path.basename(filepath)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Case 1: File contains 'summary' or report format
        if isinstance(data, dict) and "summary" in data:
            summary = data["summary"]
            # Extract first top-level model key if nested
            if summary and isinstance(summary, dict):
                first_key = list(summary.keys())[0]
                stats = summary[first_key]
                if isinstance(stats, dict) and "total" in stats:
                    cor = stats.get("correct", 0)
                    tot = stats.get("total", 0)
                    acc = (cor / tot * 100) if tot > 0 else 0
                    return filename, "Summary Report", cor, tot, acc

        # Case 2: File contains 'log' array
        logs = []
        if isinstance(data, dict) and "log" in data:
            logs = data["log"]
        elif isinstance(data, list):
            logs = data

        if logs:
            tot = len(logs)
            cor = 0
            for item in logs:
                # Compare predicted vs correct key
                pred = item.get("llm_only") or item.get("prediction") or item.get("llm_prediction")
                gt = item.get("correct") or item.get("ground_truth")
                if pred is not None and gt is not None and str(pred).strip().upper() == str(gt).strip().upper():
                    cor += 1
            acc = (cor / tot * 100) if tot > 0 else 0
            return filename, "Log File", cor, tot, acc

        return filename, "Unknown Format", 0, 0, 0.0

    except Exception as e:
        return filename, f"Error ({e})", 0, 0, 0.0

def main():
    json_files = sorted(glob.glob("eval_*.json"))
    if not json_files:
        print("No eval_*.json files found in the current directory.")
        return

    print("=" * 75)
    print(f"{'Filename':<30} | {'Type':<16} | {'Score':<10} | {'Accuracy'}")
    print("=" * 75)

    for file in json_files:
        fname, ftype, cor, tot, acc = analyze_json(file)
        if tot > 0:
            print(f"{fname:<30} | {ftype:<16} | {cor}/{tot:<7} | {acc:.2f}%")
        else:
            print(f"{fname:<30} | {ftype:<16} | N/A        | N/A")

    print("=" * 75)

if __name__ == "__main__":
    main()
