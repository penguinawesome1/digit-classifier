import numpy as np

class RBFNetwork:
    def __init__(self, centers, sigma):
        self.centers = centers
        self.sigma = sigma

    def gaussian_kernel(self, x, center):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * self.sigma ** 2))

    def fit(self, X, y):
        self.weights = np.linalg.pinv(self.calculate_phi(X)) @ y

    def calculate_phi(self, X):
        phi = np.zeros((X.shape[0], self.centers.shape[0]))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centers):
                phi[i, j] = self.gaussian_kernel(x, center)
        return phi

    def predict(self, X):
        phi = self.calculate_phi(X)
        return phi @ self.weights

# Example usage:
centers = np.array([[0, 0], [1, 1]])
sigma = 0.5
rbf_network = RBFNetwork(centers, sigma)
rbf_network.fit(X_train, y_train)
y_pred = rbf_network.predict(X_test)
