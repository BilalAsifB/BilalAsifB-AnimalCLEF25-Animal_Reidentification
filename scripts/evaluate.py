import pandas as pd
from evaluation.metrics import evaluate_predictions

def create_submission(query_dataset, predictions, submission_path):
    sample_submission = pd.read_csv(os.path.join(query_dataset.metadata_dir, 'sample_submission.csv'))
    submission = sample_submission.copy()
    submission["Id"] = [predictions[i] if i < len(predictions) else "new_individual" for i in range(len(sample_submission))]
    submission.to_csv(submission_path, index=False)
    print(f"✅ Submission file saved as {submission_path}.")