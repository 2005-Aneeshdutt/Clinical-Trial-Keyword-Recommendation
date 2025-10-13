#!/usr/bin/env python3
"""
Simple script to show model outputs and performance metrics.
"""

import pickle
import os
import glob
from ml.data_set_splitter import DataSetSplitter
from ml.mse_tester import MseTester
from ml.average_precision_k_tester import AveragePrecisionKTester

def show_model_performance():
    """Show performance metrics for all trained models."""
    print("=== Model Performance Summary ===\n")
    
    # Find all model files
    model_files = glob.glob('./output/model*.pickle')
    if not model_files:
        print("No trained models found in output/ directory.")
        return
    
    # Load test data
    test_data = DataSetSplitter.get_test_utility_matrix('./output/fullAllPublicXML.sds')
    
    for model_file in sorted(model_files):
        print(f"Model: {os.path.basename(model_file)}")
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Calculate MSE
            mse_tester = MseTester(model)
            mse = mse_tester(test_data)
            
            # Calculate AP@K
            apk_tester = AveragePrecisionKTester(model)
            apk = apk_tester(test_data)
            
            print(f"  MSE: {mse:.6f}")
            print(f"  AP@K: {apk:.6f}")
            
            # Show model parameters if available
            if hasattr(model, '_alpha'):
                print(f"  Alpha: {model._alpha}")
            if hasattr(model, '_lambda'):
                print(f"  Lambda: {model._lambda}")
            if hasattr(model, '_k'):
                print(f"  K: {model._k}")
            
            print()
            
        except Exception as e:
            print(f"  Error loading model: {e}")
            print()

def show_data_statistics():
    """Show statistics about the processed data."""
    print("=== Data Statistics ===\n")
    
    try:
        # Load the mesh counter to get data stats
        with open('./output/AllPublicXML.ctdmc', 'rb') as f:
            ctdmc = pickle.load(f)
        
        print(f"Total documents processed: {ctdmc.num_processed_docs()}")
        
        # Count MeSH terms
        mesh_count = sum(1 for counter in ctdmc if counter is not None)
        print(f"Total MeSH terms: {mesh_count}")
        
        # Show some sample statistics
        print(f"Average MeSH terms per document: {mesh_count / ctdmc.num_processed_docs():.2f}")
        
    except Exception as e:
        print(f"Error loading data statistics: {e}")

def show_output_files():
    """List all output files and their purposes."""
    print("=== Output Files ===\n")
    
    output_files = [
        ("AllPublicXML.ctdmc", "MeSH term counts from clinical trials"),
        ("fullAllPublicXML.sds", "Split dataset (train/validate/test)"),
        ("model-*.pickle", "Trained models with different parameters"),
    ]
    
    for pattern, description in output_files:
        if pattern.endswith('*'):
            files = glob.glob(f'./output/{pattern}')
            if files:
                print(f"{pattern}: {description}")
                for file in files:
                    size = os.path.getsize(file) / (1024*1024)  # MB
                    print(f"  - {os.path.basename(file)} ({size:.1f} MB)")
            else:
                print(f"{pattern}: No files found")
        else:
            if os.path.exists(f'./output/{pattern}'):
                size = os.path.getsize(f'./output/{pattern}') / (1024*1024)
                print(f"{pattern}: {description} ({size:.1f} MB)")
            else:
                print(f"{pattern}: Not found")
    
    print()
    
    # Results files
    results_files = [
        ("finding_alpha.pdf", "Training loss curves for different alpha values"),
        ("finding_params.pdf", "Parameter search analysis"),
        ("stats_of_train_data.pdf", "Training dataset statistics"),
    ]
    
    print("Results files:")
    for filename, description in results_files:
        if os.path.exists(f'./results/{filename}'):
            size = os.path.getsize(f'./results/{filename}') / 1024  # KB
            print(f"  {filename}: {description} ({size:.1f} KB)")
        else:
            print(f"  {filename}: Not found")

if __name__ == '__main__':
    show_output_files()
    show_data_statistics()
    show_model_performance()
