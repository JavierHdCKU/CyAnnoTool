import subprocess
from pathlib import Path

def run(input_files, output_files, params, **kwargs):
    matrix = input_files["data.matrix"]
    labels = input_files["data.true_labels"]

    pred_path = Path(output_files["analysis.prediction.cyannotool"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(Path(__file__).resolve().parents[1] / "run_cyanno.py"),
        str(matrix),
        str(labels),
        str(pred_path)  # FIX
    ]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
