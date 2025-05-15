def compute_baus(query_labels, query_species, db_labels, db_species, predictions, distance_matrix, threshold=0.6):
    baus = 0
    for i, (true_label, pred_idx) in enumerate(zip(query_labels, predictions)):
        if db_labels[pred_idx] == true_label and db_species[pred_idx] == query_species[i] and distance_matrix[i, 0] < threshold:
            baus += 1
    return baus / len(query_labels) if len(query_labels) > 0 else 0