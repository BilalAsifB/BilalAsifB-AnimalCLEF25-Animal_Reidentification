import torch
from scripts.evaluate import create_submission
from evaluation.metrics import evaluate_predictions
import pandas as pd
import numpy as np

def one_shot_evaluation(model, database_loader, query_loader, query_dataset, config):
    """Evaluate Siamese Network using one-shot learning (CPU-only)."""
    model.eval()
    identity_to_embedding = {}
    database_identities = set()
    
    with torch.no_grad():
        for img, label, species, _ in database_loader:
            embedding = model.forward_one(img)
            if label != 'new_individual':
                if label not in identity_to_embedding:
                    identity_to_embedding[label] = embedding
                database_identities.add(label)
    
    solution_data = []
    submission_data = []
    predictions = []
    with torch.no_grad():
        for img, true_label, species, image_id in query_loader:
            query_embedding = model.forward_one(img)
            
            min_distance = float('inf')
            predicted_label = 'new_individual'
            for db_label, db_embedding in identity_to_embedding.items():
                distance = torch.norm(query_embedding - db_embedding, p=2, dim=1).item()
                if distance < min_distance:
                    min_distance = distance
                    predicted_label = db_label if distance < config['threshold'] else 'new_individual'
            
            solution_data.append({
                'image_id': image_id,
                'identity': true_label,
                'dataset': species,
                'new_identity': true_label not in database_identities
            })
            submission_data.append({
                'image_id': image_id,
                'identity': predicted_label
            })
            predictions.append(predicted_label)
    
    create_submission(query_dataset, predictions, config['submission_path'])
    
    final_score = evaluate_predictions(solution_data, submission_data)
    return final_score