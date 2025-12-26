from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Train:
    def __init__(self, X, y, model):
        self.model = model
        self.X = X
        self.y = y

    def train(self, random_state=42, loocv=False, test_size=0.2, out_path=None, **hyperparams):
        """ Train the model with optional Leave-One-Out cross-validation """
        print(
            f"🚀 Training model: {self.model.__name__}\n"
            f"  Random state: {random_state}\n"
            f"  Leave-One-Out CV: {loocv}"
        )
        if loocv:
            return self._train_loocv(random_state, out_path, **hyperparams)
        else:
            print(f"  Test size: {test_size}\n")
            return self._train_baseline(test_size, random_state, out_path, **hyperparams)
    

    def _train_baseline(self, test_size, random_state, out_path, **hyperparams):
        """ Train with a standard train-test split """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        # Scale the features
        X_train_scaled, X_test_scaled = self._scale(X_train, X_test)
        # Fit the model
        model = self._fit(X_train_scaled, y_train, random_state, **hyperparams)
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        # Probabilities
        y_proba = model.predict_proba(X_test_scaled)[:, 1] 
        
        print(f"✅ Training complete.\n")

        results = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba
        }

        # Save if output path is provided
        self._save_results(results, out_path)
        
        return results

    def _train_loocv(self, random_state, out_path, **hyperparams):
        
        """ Train using Leave-One-Out cross-validation """
        from sklearn.model_selection import LeaveOneOut
        loocv = LeaveOneOut()
        y_tests = []
        y_preds = []
        y_probas = []
        
        for train_index, test_index in loocv.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            # Scale the features
            X_train_scaled, X_test_scaled = self._scale(X_train, X_test)
            # Fit the model
            model = self._fit(X_train_scaled, y_train, random_state, **hyperparams)
            # Predict on the test sample
            y_pred = model.predict(X_test_scaled)
            # Probabilities
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            # Store results
            y_tests.append(y_test.values[0])
            y_preds.append(y_pred[0])
            y_probas.append(y_proba[0])
        
        print(f"✅ Training complete.\n")

        results = {
            "model": model,
            "y_test": y_tests,
            "y_pred": y_preds,
            "y_proba": y_probas,
        }

        # Save if output path is provided
        self._save_results(results, out_path)

        return results

    def _scale(self, X_train, X_test):
        """ Scale features using StandardScaler """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def _fit(self, X_train, y_train, random_state, **hyperparams):
        """ Fit the model """
        model = self.model(random_state=random_state, **hyperparams)
        model.fit(X_train, y_train)
        return model    

    def _save_results(self, results, out_path):
        import joblib
        if out_path:
            # Ensure directory exists
            out_path.mkdir(parents=True, exist_ok=True)
            # Save to file inside directory
            file_path = out_path / "results.pkl"
            joblib.dump(results, file_path)
            print(f"✅ Results saved to {file_path}")