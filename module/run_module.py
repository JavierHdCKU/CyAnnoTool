matrix = input_files["data.matrix"]
labels = input_files["data.true_labels"]
pred_path = Path(output_files["analysis.prediction.cyannotool"])
pred_path.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    "python",
    str(Path(__file__).resolve().parents[1] / "run_cyanno.py"),
    str(matrix),
    str(labels),
    str(pred_path)  # note: file, not directory
]
subprocess.check_call(cmd)
