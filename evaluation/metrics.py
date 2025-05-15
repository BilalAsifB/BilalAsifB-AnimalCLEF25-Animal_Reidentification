from .compute_baks import compute_baks
from .compute_baus import compute_baus

def evaluate_predictions(solution_data, submission_data):
    query_labels = [item['identity'] for item in solution_data]
    query_species = [item['dataset'] for item in solution_data]
    db_labels = [item['identity'] for item in submission_data]
    db_species = [item['dataset'] for item in submission_data]
    predictions = [db_labels.index(item['identity']) for item in submission_data]
    distance_matrix = np.array([[0.5] * len(predictions)] * len(predictions))  # Placeholder distance
    baks = compute_baks(query_labels, query_species, db_labels, db_species, predictions)
    baus = compute_baus(query_labels, query_species, db_labels, db_species, predictions, distance_matrix)
    geometric_mean = (baks * baus) ** 0.5 if baks > 0 and baus > 0 else 0
    return {"BAKS": baks, "BAUS": baus, "Geometric Mean": geometric_mean}