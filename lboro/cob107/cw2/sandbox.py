class SingleLayerAnn:
    def __init__(self, layers):
        self.W1 = np.random.standard_normal((layers[0], layers[1]))/layers[0]**0.5
        self.W2 = np.random.standard_normal((layers[1], layers[2]))/layers[1]**0.5
        self.B1 = np.random.randn(layers[1],1) / layers[0] ** 0.5
        self.B2 = np.random.randn(layers[2], 1) / layers[1] ** 0.5
        
    def activation(self, x, func_name="sigmoid"):
        if func_name == "sigmoid":
            return 1/(1+np.exp(-x))
    
    def activation_deriv(self, a, func_name="sigmoid"):
        if func_name == "sigmoid":
            return a * (1 - a)
    
    def train(self, features, targets):
        results = pd.DataFrame(columns=["mse", "st_mse"])
        real_targets = unstandardise_value(targets, max_val, min_val)
        for _ in range(2000):
            # Forward pass
            ## Hidden layer
            A1 = self.activation(np.dot(features, self.W1) + self.B1.T)
            ## Output Layer
            A2 = self.activation(np.dot(A1, self.W2) + self.B2)

            # Error calculation
            real_preds = unstandardise_value(A2, max_val, min_val)
            results = results.append({
                "mse": mean_squared_error(real_targets, real_preds),
                "st_mse": mean_squared_error(targets, A2),
                "mae": mean_absolute_error(real_targets, real_preds), 
                "st_mae": mean_absolute_error(targets, A2)
            }, ignore_index=True)

            # Backward pass
            E1 = targets - A2
            dw1 = E1 * self.activation_deriv(A2)

            E2 = np.dot(dw1, self.W2.T)
            dw2 = E2 * self.activation_deriv(A1)

            self.W2 += 0.1 * (np.dot(A1.T, dw1)) / len(targets)
            self.W1 += 0.1 * (np.dot(features.T, dw2)) / len(targets)
            self.B2 += 0.1 * np.dot(dw1.T, np.ones((len(targets), 1))) / len(targets)
            self.B1 += 0.1 * np.dot(dw2.T, np.ones((len(targets), 1))) / len(targets)
        
        return results