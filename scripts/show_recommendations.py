#!/usr/bin/env python3
"""
Script to show keyword recommendations from the trained model.
This will load a trained model and show relevant MeSH terms for a given document.
"""

import pickle
import numpy as np
from ml.data_set_splitter import DataSetSplitter
from clinical_trials.clinical_trial_document_mesh_counter import ClinicalTrialDocumentMeshCounter
import sys
import os
import glob

def get_latest_model_pickle(directory='./output/'):
    """Find the newest model pickle file."""
    list_of_files = glob.glob(os.path.join(directory, 'model*.pickle'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def load_mesh_terms():
    """Load MeSH terms from the counter."""
    with open('./output/AllPublicXML.ctdmc', 'rb') as f:
        ctdmc = pickle.load(f)
    return ctdmc

def show_recommendations(model, ctdmc, doc_index=0, top_k=10):
    """Show top-k MeSH term recommendations for a document."""
    print(f"\n=== MeSH Term Recommendations for Document {doc_index} ===")
    
    # Get the document's current MeSH terms
    current_terms = []
    for mesh_index, counter in enumerate(ctdmc):
        if counter is not None and counter[doc_index] > 0:
            current_terms.append((mesh_index, counter[doc_index]))
    
    print(f"\nCurrent MeSH terms in document {doc_index}:")
    for mesh_index, tfidf in sorted(current_terms, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - MeSH Index {mesh_index}: TF-IDF = {tfidf:.4f}")
    
    # Get predictions for this document
    # Get the total number of MeSH terms
    total_mesh_terms = sum(1 for counter in ctdmc if counter is not None)
    doc_vector = np.zeros(total_mesh_terms)
    
    # Map mesh indices to vector positions
    mesh_index_map = []
    for mesh_index, counter in enumerate(ctdmc):
        if counter is not None:
            mesh_index_map.append(mesh_index)
    
    # Fill the document vector
    for i, mesh_index in enumerate(mesh_index_map):
        counter = ctdmc._counter[mesh_index]
        if counter is not None:
            doc_vector[i] = counter[doc_index]
    
    predictions = model.predict(doc_vector)
    
    # Get top recommendations (excluding already present terms)
    recommendations = []
    for i, score in enumerate(predictions):
        mesh_index = mesh_index_map[i]
        counter = ctdmc._counter[mesh_index]
        if counter is not None and counter[doc_index] == 0:  # Not already in document
            recommendations.append((mesh_index, score))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_k} recommended MeSH terms:")
    for i, (mesh_index, score) in enumerate(recommendations[:top_k]):
        print(f"  {i+1:2d}. MeSH Index {mesh_index}: Score = {score:.4f}")
    
    return recommendations[:top_k]

def show_model_info(model_path):
    """Show information about the loaded model."""
    print(f"=== Model Information ===")
    print(f"Model file: {model_path}")
    print(f"Model type: {type(model).__name__}")
    
    # Try to get model parameters
    if hasattr(model, '_alpha'):
        print(f"Alpha (learning rate): {model._alpha}")
    if hasattr(model, '_lambda'):
        print(f"Lambda (regularization): {model._lambda}")
    if hasattr(model, '_k'):
        print(f"K (latent factors): {model._k}")
    if hasattr(model, '_tau'):
        print(f"Tau (threshold): {model._tau}")

if __name__ == '__main__':
    # Load the latest model
    model_path = get_latest_model_pickle()
    if model_path is None:
        print("No model pickle found in the output directory.")
        print("Please train a model first by running: python scripts/search_alpha.py")
        sys.exit(1)
    
    print(f"Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load MeSH terms
    print("Loading MeSH terms...")
    ctdmc = load_mesh_terms()
    
    # Show model info
    show_model_info(model_path)
    
    # Show recommendations for first few documents
    num_docs = min(3, ctdmc.num_processed_docs())
    for doc_idx in range(num_docs):
        show_recommendations(model, ctdmc, doc_idx, top_k=10)
        print("\n" + "="*60)
    
    print(f"\nTotal documents in dataset: {ctdmc.num_processed_docs()}")
    print(f"Total MeSH terms: {len([c for c in ctdmc if c is not None])}")
