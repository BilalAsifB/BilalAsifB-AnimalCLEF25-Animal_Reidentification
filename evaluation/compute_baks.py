def compute_baks(query_labels, query_species, db_labels, db_species, predictions):
    baks = 0
    for i, (true_label, pred_idx) in enumerate(zip(query_labels, predictions)):
        if db_labels[pred_idx] == true_label and db_species[pred_idx] == query_species[i]:
            baks += 1
    return baks / len(query_labels) if len(query_labels) > 0 else 0