import subprocess
from pathlib import Path
import shutil

def run(input_files, output_files, params, **kwargs):
    matrix = input_files["data.matrix"]
    labels = input_files["data.true_labels"]

    pred_path = Path(output_files["analysis.prediction.cyannotool"])
    output_dir = pred_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run script
    cmd = [
        "python",
        str(Path(__file__).resolve().parents[1] / "run_cyanno.py"),
        str(matrix),
        str(labels),
        str(output_dir)
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    # Move output to expected location
    generated = output_dir / "output_predictions.txt"
    shutil.move(generated, pred_path)
