#!/usr/bin/env python3
"""
Final working script to show recommendations without array comparison issues.
"""

import pickle
import numpy as np
from ml.data_set_splitter import DataSetSplitter
import os
import glob

def get_latest_model_pickle(directory='./output/'):
    """Find the newest model pickle file."""
    list_of_files = glob.glob(os.path.join(directory, 'model*.pickle'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def show_final_recommendations():
    """Show recommendations using a simpler approach."""
    print("=== XLINIQ Final Recommendations ===\n")
    
    # Load the latest model
    model_path = get_latest_model_pickle()
    if model_path is None:
        print("No model pickle found in the output directory.")
        return
    
    print(f"Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model type: {type(model).__name__}")
    if hasattr(model, '_alpha'):
        print(f"Learning rate (alpha): {model._alpha}")
    if hasattr(model, '_lambda'):
        print(f"Regularization (lambda): {model._lambda}")
    if hasattr(model, '_k'):
        print(f"Latent factors (k): {model._k}")
    
    # Load the split dataset
    print("\nLoading split dataset...")
    try:
        train_data = DataSetSplitter.get_train_utility_matrix('./output/fullAllPublicXML.sds')
        val_data = DataSetSplitter.get_validate_utility_matrix('./output/fullAllPublicXML.sds')
        test_data = DataSetSplitter.get_test_utility_matrix('./output/fullAllPublicXML.sds')
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Validation data shape: {val_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Show some sample recommendations
        print("\n=== Sample Recommendations ===")
        
        # Get a sample document from test set
        sample_doc_idx = 0
        sample_doc = test_data[:, sample_doc_idx].toarray().flatten()
        
        print(f"\nDocument {sample_doc_idx} current MeSH terms:")
        current_terms = []
        for i, score in enumerate(sample_doc):
            if score > 0:
                current_terms.append((i, score))
        
        # Show top current terms
        current_terms.sort(key=lambda x: x[1], reverse=True)
        for i, (mesh_idx, score) in enumerate(current_terms[:5]):
            print(f"  {i+1}. MeSH Index {mesh_idx}: TF-IDF = {score:.4f}")
        
        # Get predictions
        print("\nGetting predictions...")
        predictions = model.predict(sample_doc)
        
        # Flatten predictions if it's 2D
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        
        # Show ALL predictions (not filtering)
        print(f"\nTop 10 MeSH term predictions (all terms):")
        all_predictions = [(i, float(score)) for i, score in enumerate(predictions)]
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        
        for i, (mesh_idx, score) in enumerate(all_predictions[:10]):
            print(f"  {i+1:2d}. MeSH Index {mesh_idx}: Score = {score:.4f}")
        
        # Show model performance
        print(f"\n=== Model Performance ===")
        from ml.mse_tester import MseTester
        from ml.average_precision_k_tester import AveragePrecisionKTester
        
        print("Calculating MSE...")
        mse_tester = MseTester(model)
        mse = mse_tester(test_data)
        # Handle tuple result
        if isinstance(mse, tuple):
            mse = mse[0]
        print(f"MSE on test set: {mse:.6f}")
        
        print("Calculating AP@K...")
        apk_tester = AveragePrecisionKTester(model)
        apk = apk_tester(test_data)
        # Handle tuple result
        if isinstance(apk, tuple):
            apk = apk[0]
        print(f"AP@K on test set: {apk:.6f}")
        
        print(f"\n=== Summary ===")
        print(f"✅ Model successfully loaded and tested")
        print(f"✅ Dataset contains {test_data.shape[1]} documents")
        print(f"✅ Model uses {test_data.shape[0]} MeSH terms")
        print(f"✅ Performance: MSE={mse:.4f}, AP@K={apk:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    show_final_recommendations()
