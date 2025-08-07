import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from scipy.optimize import minimize

from charts import plot_frontier


class SkewedTOptimizer:
    """
    Mean-Variance Optimization using CVXPY
    
    The optimization problem:
    min (1/2) * w^T * Sigma * w
    subject to: w^T * mu = target_return (if specified)
                w^T * 1 = 1 (weights sum to 1)
    
    Solution uses CVXPY optimizer    
    """

    def __init__(self, returns_df):
        """
        Initialize with returns data
        
        Parameters:
        returns_df: pandas DataFrame with returns (rows=periods, cols=assets)
        """

        # Annualize returns
        periods_per_year = 1
        returns_df = returns_df * periods_per_year

        self.returns = returns_df.to_numpy()

        self.calculate_moments()

        self.asset_risk_data = np.zeros(self.num_assets)
        self.asset_return_data = np.zeros(self.num_assets)

        for i in range(self.num_assets):
            self.asset_risk_data[i] = np.sqrt(self.cov[i, i])
            self.asset_return_data[i] = self.mean[i]

        print(f"Portfolio initalized with {self.num_assets} assets")

        self.initial_weights = np.ones(self.num_assets) / self.num_assets
        self.weights = np.ones(self.num_assets) / self.num_assets

        return

    def calculate_moments(self):
        """
        Calculate the moments for the returns data distribution.
        """

        self.num_assets = self.returns.shape[1]
        self.mean = np.mean(self.returns, axis=0)
        self.cov = np.cov(self.returns.T)
        centered_returns = self.returns - self.mean

        self.co_skewness = np.zeros((self.num_assets, self.num_assets, self.num_assets))
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                for k in range(self.num_assets):
                    self.co_skewness[i, j, k] = np.mean(centered_returns[:, i] * centered_returns[:, j] * centered_returns[:, k])        

        self.co_kurtosis = np.zeros((self.num_assets, self.num_assets, self.num_assets, self.num_assets))
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                for k in range(self.num_assets):
                    for l in range(self.num_assets):
                        self.co_kurtosis[i, j, k, l] = np.mean(centered_returns[:, i] * centered_returns[:, j] * centered_returns[:, k] * centered_returns[:, l])

        # print("Number of assets:", self.num_assets)
        # print("Mean Vector:\n", self.mean)
        # print("\nCovariance Matrix:\n", self.cov)

        # Note: Printing the full co-skewness and co-kurtosis matrices would be very large,
        # so we'll just show their shape to demonstrate they've been created.
        # print("\nCo-skewness matrix shape:", self.co_skewness.shape)
        # print("Co-kurtosis matrix shape:", self.co_kurtosis.shape)

        return



    def objective_function(self, weights, A, B, C):
        """
        Calculates a portfolio objective function with higher order moments.
        """
        mean_term = weights.T @ self.mean
        # variance_term = weights.T @ cov @ weights
        variance_term = np.dot(weights.flatten(), self.cov @ weights.flatten())
        # skewness_term = weights.T @ co_skewness @ (weights * weights)
        skewness_term = np.einsum('i,j,k,ijk->', weights.flatten(), weights.flatten(), weights.flatten(), self.co_skewness)
        # kurtosis_term = weights.T @ co_kurtosis @ (weights * weights * weights)
        kurtosis_term = np.einsum('i,j,k,l,ijkl->', weights.flatten(), weights.flatten(), weights.flatten(), weights.flatten(), self.co_kurtosis)

        return (-mean_term + A / 2 * variance_term - B / 6 * skewness_term + C / 24 * kurtosis_term)


    def run_optimization(self, A, B, C):
        """
        Runs the optimization for a single portfolio.
        """
        objective_args = (A, B, C)
        # objective_args = (self.mean, self.cov, self.co_skewness, self.co_kurtosis, A, B, C)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))

        options = {'disp': True, 'maxiter': 500}
        result = minimize(
            self.objective_function,
            self.initial_weights,
            args = objective_args,
            method = 'SLSQP',
            bounds = bounds,
            constraints = constraints,
            options = options
        )

        optimal_weights = np.zeros(self.num_assets)
        expected_return = 0
        variance = 0

        # Print the results
        if result.success:
            optimal_weights = result.x.reshape(-1, 1)
            optimal_value = result.fun
            expected_return = (optimal_weights.T @ self.mean).item()
            variance = (optimal_weights.T @ self.cov @ optimal_weights).item()
            print(f"Optimization successful!")
            print("\nOptimal Weights:")
            for i in range(self.num_assets):
                print(f"Asset {i}: {optimal_weights[i][0]:.4f}")
            print(f"\nMinimum objective function value: {optimal_value:.6f}")
        else:
            print("Optimization failed.")
            print(result.message)

            # portfolios.append({
            #     'gamma': gamma_vals[i],
            #     'return': r_rar,
            #     'variance': v_rar,
            #     'std_dev': np.sqrt(v_rar),
            #     'weights': w_rar
            #     })

        return optimal_weights, expected_return, variance


    def efficient_frontier_skewed_t(self, A, B, C, num_points=100):
        """
        Generate efficient frontier using skewed-t cost function

        Parameters:
        num_points: number of points on the efficient frontier
        
        Returns:
        portfolio
        """
        print(f"\n" + "="*50)
        print("EFFICIENT FRONTIER")
        print("="*50)
    
        gamma_vals = np.logspace(-2, 3, num=num_points)      
        
        portfolios = []

        for i in range(num_points):
            w_st, r_st, v_st = self.run_optimization(gamma_vals[i], B, C)

            portfolios.append({
                'A': gamma_vals[i],
                'return': r_st,
                'variance': v_st,
                'std_dev': np.sqrt(v_st),
                'weights': w_st
                })
       
        print(f"Generated {len(portfolios)} efficient portfolios")

        return portfolios      



if __name__ == "__main__":
    # Load the CSV file back into a dataframe
    returns_df = pd.read_csv('./data/portfolio_returns.csv', index_col=0)

    # Initialize optimizer
    optimizer = SkewedTOptimizer(returns_df)

    weights = np.ones(25) / 25

    A = 2
    B = -1
    C = -1

    print(optimizer.objective_function(weights, A, B, C))

    w_st, r_st, v_st = optimizer.run_optimization(A, B, C)

    print(f"Mean: {r_st:.5f}")
    print(f"Variance: {v_st:.5f}")
    print(f"Standard Deviation: {np.sqrt(v_st):.5f}")
    # # Global minimum variance portfolio
    # w_mv, r_mv, v_mv = optimizer.minimum_variance_portfolio()
    # # print(w_mv, r_mv, v_mv)

    # w_rar, r_rar, v_rar = optimizer.risk_adjusted_return_portfolio(gamma_value=1.0)
    # # print(w_mv, r_mv, v_mv)

    SAMPLES = 100
    # Generate and plot efficient frontier
    portfolios = optimizer.efficient_frontier_skewed_t(A, B, C, num_points=50)

    efficient_frontier_risk_data = [p['std_dev'] for p in portfolios]
    efficient_frontier_return_data = [p['return'] for p in portfolios]

    asset_risk_data = np.sqrt(np.diag(optimizer.cov))
    asset_return_data = optimizer.mean
    asset_labels = [col for col in returns_df.columns]

    # point_risk_data = [np.sqrt(v_mv)]
    # point_return_data = [r_mv]
    # point_labels = ["Minimum variance portfolio", "Tangent portfolio"]

    plot_frontier(efficient_frontier_risk_data, efficient_frontier_return_data, asset_risk_data, asset_return_data, asset_labels, annualize_data=True)
