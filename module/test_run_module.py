import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from module.run_module import run


input_files = {
    "data.matrix": "/home/javier/ob-pipeline-cytof/out/data/data_import/.39e04c32bfe741dadaaa3fc7842586f32079c7042a399c9ffb6316987a4d6666/preprocessing/data_preprocessing/default/out/data/data_import/.39e04c32bfe741dadaaa3fc7842586f32079c7042a399c9ffb6316987a4d6666/preprocessing/data_preprocessing/default/data_import.matrix.gz",
    "data.true_labels": "/home/javier/ob-pipeline-cytof/out/data/data_import/.39e04c32bfe741dadaaa3fc7842586f32079c7042a399c9ffb6316987a4d6666/preprocessing/data_preprocessing/default/out/data/data_import/.39e04c32bfe741dadaaa3fc7842586f32079c7042a399c9ffb6316987a4d6666/preprocessing/data_preprocessing/default/data_import.true_labels.gz"
}

output_files = {
    "analysis.prediction.cyannotool": "test_predictions_CyAnno.txt"
}

params = {}

run(input_files, output_files, params)
print("Module executed OK.")
