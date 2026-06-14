import joblib, numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class DummyModel(BaseEstimator, ClassifierMixin):
    def predict_proba(self, X):
        # Return equal probability for both classes
        probs = np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])
        return probs

# Save dummy model
joblib.dump(DummyModel(), r'F:\cliniguard\final_website\cliniguard_model.joblib')
print('Dummy model created')
