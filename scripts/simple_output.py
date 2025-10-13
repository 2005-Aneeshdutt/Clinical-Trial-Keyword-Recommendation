#!/usr/bin/env python3
"""
Simple script to show basic outputs and model information.
"""

import pickle
import os
import glob

def main():
    print("=== XLINIQ Project Outputs ===\n")
    
    # Check if we're in the right directory
    if not os.path.exists('./output'):
        print("Error: output directory not found. Please run from alphatesting directory.")
        return
    
    # List output files
    print("1. OUTPUT FILES:")
    output_files = os.listdir('./output')
    for file in sorted(output_files):
        if file.endswith('.pickle'):
            size = os.path.getsize(f'./output/{file}') / (1024*1024)
            print(f"   📁 {file} ({size:.1f} MB) - Trained model")
        elif file.endswith('.ctdmc'):
            size = os.path.getsize(f'./output/{file}') / (1024*1024)
            print(f"   📊 {file} ({size:.1f} MB) - MeSH term counts")
        elif file.endswith('.sds'):
            size = os.path.getsize(f'./output/{file}') / (1024*1024)
            print(f"   📈 {file} ({size:.1f} MB) - Split dataset")
    
    print("\n2. RESULTS FILES:")
    if os.path.exists('./results'):
        result_files = os.listdir('./results')
        for file in sorted(result_files):
            if file.endswith('.pdf'):
                size = os.path.getsize(f'./results/{file}') / 1024
                print(f"   📄 {file} ({size:.1f} KB) - Analysis plot")
    
    print("\n3. MODEL PERFORMANCE:")
    try:
        # Load the latest model
        model_files = glob.glob('./output/model*.pickle')
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            print(f"   Latest model: {os.path.basename(latest_model)}")
            
            with open(latest_model, 'rb') as f:
                model = pickle.load(f)
            
            print(f"   Model type: {type(model).__name__}")
            if hasattr(model, '_alpha'):
                print(f"   Learning rate (alpha): {model._alpha}")
            if hasattr(model, '_lambda'):
                print(f"   Regularization (lambda): {model._lambda}")
            if hasattr(model, '_k'):
                print(f"   Latent factors (k): {model._k}")
            if hasattr(model, '_tau'):
                print(f"   Threshold (tau): {model._tau}")
        else:
            print("   No trained models found.")
    except Exception as e:
        print(f"   Error loading model: {e}")
    
    print("\n4. WHAT THIS PROJECT DOES:")
    print("   🎯 Purpose: Recommend MeSH terms (medical keywords) for clinical trials")
    print("   🔬 Method: Collaborative filtering with ReLU activation")
    print("   📚 Data: Clinical trial documents from ClinicalTrials.gov")
    print("   🧠 Algorithm: Mini-batch gradient descent with latent factors")
    
    print("\n5. HOW TO USE:")
    print("   📖 View plots: Open PDF files in results/ folder")
    print("   🔍 Test model: python scripts/test_final_model.py")
    print("   📊 See recommendations: python scripts/show_recommendations.py")
    print("   📈 Analyze results: python analysis/analyze_alpha_search.py")

if __name__ == '__main__':
    main()
