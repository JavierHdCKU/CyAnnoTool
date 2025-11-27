#!/usr/bin/env python3
"""
Generate HTML performance report for CyAnno results
"""

import json
import pandas as pd
from pathlib import Path
import base64

def main():
    # Get inputs and outputs from snakemake
    predictions_file = snakemake.input.predictions
    metrics_file = snakemake.input.metrics
    training_metrics_file = snakemake.input.training_metrics
    confusion_matrix_file = snakemake.input.confusion_matrix
    output_file = snakemake.output.report
    
    # Load data
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    with open(training_metrics_file, 'r') as f:
        training_metrics = json.load(f)
    
    predictions_data = pd.read_csv(predictions_file)
    
    # Encode confusion matrix image
    with open(confusion_matrix_file, 'rb') as f:
        cm_encoded = base64.b64encode(f.read()).decode()
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CyAnno Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 8px; }}
            .metric-box {{ background-color: #e8f4fd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .cell-type {{ margin: 5px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .confusion-matrix {{ text-align: center; margin: 20px 0; }}
            .confusion-matrix img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>CyAnno Cell Type Annotation Report</h1>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <h2>Summary</h2>
        <div class="metric-box">
            <h3>Overall Performance</h3>
            <p><strong>Overall Accuracy:</strong> {metrics['overall_accuracy']:.3f}</p>
            <p><strong>F1 Score (Weighted):</strong> {metrics['f1_weighted']:.3f}</p>
            <p><strong>F1 Score (Macro):</strong> {metrics['f1_macro']:.3f}</p>
            <p><strong>Total Samples Evaluated:</strong> {metrics['n_samples']}</p>
        </div>

        <div class="metric-box">
            <h3>Training Performance</h3>
            <p><strong>Training F1 Score:</strong> {training_metrics['f1_score']:.3f}</p>
            <p><strong>Training Samples:</strong> {training_metrics['n_training_samples']}</p>
            <p><strong>Validation Samples:</strong> {training_metrics['n_validation_samples']}</p>
        </div>

        <h2>Per-Cell Type Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Cell Type</th>
                    <th>F1 Score</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Support</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add per-cell type metrics
    class_report = metrics['classification_report']
    for cell_type in metrics['cell_types']:
        if cell_type in class_report:
            cr = class_report[cell_type]
            html_content += f"""
                <tr>
                    <td>{cell_type}</td>
                    <td>{metrics['f1_per_class'][cell_type]:.3f}</td>
                    <td>{metrics['accuracy_per_class'][cell_type]:.3f}</td>
                    <td>{cr['precision']:.3f}</td>
                    <td>{cr['recall']:.3f}</td>
                    <td>{cr['support']}</td>
                </tr>
            """
    
    html_content += """
            </tbody>
        </table>

        <h2>Confusion Matrix</h2>
        <div class="confusion-matrix">
    """
    
    html_content += f'<img src="data:image/png;base64,{cm_encoded}" alt="Confusion Matrix" />'
    
    html_content += """
        </div>

        <h2>Prediction Distribution</h2>
        <table>
            <thead>
                <tr>
                    <th>Predicted Cell Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add prediction distribution
    pred_counts = predictions_data['predicted_cell_type'].value_counts()
    total_preds = len(predictions_data)
    for cell_type, count in pred_counts.items():
        percentage = (count / total_preds) * 100
        html_content += f"""
            <tr>
                <td>{cell_type}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
        """
    
    html_content += """
            </tbody>
        </table>

        <h2>Model Details</h2>
        <div class="metric-box">
            <p><strong>Cell Types Detected:</strong> """ + ", ".join(metrics['cell_types']) + """</p>
            <p><strong>Algorithm:</strong> Random Forest Classifier</p>
            <p><strong>Normalization:</strong> Applied</p>
        </div>

    </body>
    </html>
    """
    
    # Save HTML report
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Performance report generated: {output_file}")

if __name__ == "__main__":
    main()
