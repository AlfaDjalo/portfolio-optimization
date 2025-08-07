import numpy as np
import pandas as pd

import cvxpy as cp

from charts import plot_frontier

class CVXPYOptimizer:
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
        periods_per_year = 252
        self.returns_df = returns_df * periods_per_year

        # Defining initial inputs
        self.mu = self.returns_df.mean().to_numpy().reshape(1, -1)
        self.sigma = self.returns_df.cov().to_numpy()
        self.n = self.mu.shape[1]

        self.asset_risk_data = np.zeros(self.n)
        self.asset_return_data = np.zeros(self.n)

        for i in range(self.n):
            self.asset_risk_data[i] = np.sqrt(self.sigma[i, i])
            self.asset_return_data[i] = self.mu[0, i]

        print(f"Portfolio initalized with {self.n} assets")

        self.weights = cp.Variable(self.n)
        self.risk = cp.quad_form(self.weights, self.sigma)
        self.ret = self.mu @ self.weights

        self.gamma = cp.Parameter(nonneg=True, value=1.0)

    def minimum_variance_portfolio(self):
        """
        Global minimum variance portfolio (analytic solution)
        
        Solution: w = (Sigma^(-1) * 1) / (1^T * Sigma^(-1) * 1)
        
        Returns:
        weights, expected_return, variance
        """

        print("\n" + "="*50)
        print("GLOBAL MINIMUM VARIANCE PORTFOLIO")
        print("="*50)

        # # Defining initial variables
        # weights = cp.Variable((self.n, 1))

        # Budget and weights constraints
        constraints = [cp.sum(self.weights) == 1,
                    self.weights >= 0,
                    self.weights <= 1]


        # Defining risk objective
        # risk = cp.quad_form(self.weights, self.sigma)
        objective = cp.Minimize(self.risk)

        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Portfolio statistics
        expected_return = (self.mu @ self.weights).value[0]
        variance = self.risk.value

        print(expected_return)
        print(variance)

        print(f"Weights: {self.weights.value}")
        print(f"Expected return: {expected_return:.6f}")
        print(f"Variance: {variance:.6f}")
        print(f"Standard deviation: {np.sqrt(variance):.6f}")

        # Verify weight constraint
        print(f"Sum of weights: {np.sum(self.weights.value):.6f} (should be 1.0)")

        return self.weights.value, expected_return, variance


    # def minimum_variance_portfolio(self):
    #     """
    #     Global minimum variance portfolio (analytic solution)
        
    #     Solution: w = (Sigma^(-1) * 1) / (1^T * Sigma^(-1) * 1)
        
    #     Returns:
    #     weights, expected_return, variance
    #     """

    #     print("\n" + "="*50)
    #     print("GLOBAL MINIMUM VARIANCE PORTFOLIO")
    #     print("="*50)

    #     # Defining initial variables
    #     self.weights = cp.Variable((self.n, 1))

    #     # Budget and weights constraints
    #     constraints = [cp.sum(self.weights) == 1,
    #                 self.weights >= 0,
    #                 self.weights <= 1]


    #     # Defining risk objective
    #     # risk = cp.quad_form(self.weights, self.sigma)
    #     objective = cp.Minimize(risk)

    #     prob = cp.Problem(objective, constraints)
    #     prob.solve()

    #     # Portfolio statistics
    #     expected_return = (self.mu @ weights).value[0][0]
    #     variance = risk.value[0][0]

    #     print(expected_return)

    #     print(f"Weights: {self.weights.value}")
    #     print(f"Expected return: {expected_return:.6f}")
    #     print(f"Variance: {variance:.6f}")
    #     print(f"Standard deviation: {np.sqrt(variance):.6f}")

    #     # Verify weight constraint
    #     print(f"Sum of weights: {np.sum(weights.value):.6f} (should be 1.0)")

    #     return self.weights.value, expected_return, variance


    def risk_adjusted_return_portfolio(self, gamma_value=1.0):
        """
        Optimal portfolio based on risk-adjusted return for a given gamma
        
        This is the constrained optimization problem:
        min (1/2) * w^T * Sigma * w
        s.t. w^T * mu = target_return
             w^T * 1 = 1
        
        Solution using Lagrange multipliers:
        w = Sigma^(-1) * (g*mu + h*1)
        where g and h are solved analytically
        
        Parameters:
        target_return: desired portfolio return
        
        Returns:
        weights, expected_return, variance
        """

        print(f"\n" + "="*50)
        print(f"MAXIMUM RISK-ADJUSTED RETURN FOR GAMMA = {gamma_value:.6f}")
        print("="*50)

        # Using risk-adjusted return as the objective
        # ret = mu @ w
        # gamma = cp.Parameter(nonneg=True)

        # Budget and weights constraints
        constraints = [cp.sum(self.weights) == 1,
                    self.weights >= 0,
                    self.weights <= 1]

        objective = cp.Maximize(self.ret - gamma_value * self.risk)

        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Portfolio statistics
        expected_return = (self.mu @ self.weights).value[0]
        variance = self.risk.value

        print(expected_return)

        print(f"Weights: {self.weights.value}")
        print(f"Expected return: {expected_return:.6f}")
        print(f"Variance: {variance:.6f}")
        print(f"Standard deviation: {np.sqrt(variance):.6f}")

        # Verify weight constraint
        print(f"Sum of weights: {np.sum(self.weights.value):.6f} (should be 1.0)")

        return self.weights.value, expected_return, variance


    def efficient_frontier_cvxpy(self, num_points=100):
        """
        Generate efficient frontier using risk-adjusted returns

        Parameters:
        num_points: number of points on the efficient frontier
        
        Returns:
        portfolio
        """
        print(f"\n" + "="*50)
        print("EFFICIENT FRONTIER")
        print("="*50)

        # Budget and weights constraints
        constraints = [cp.sum(self.weights) == 1,
                    self.weights >= 0,
                    self.weights <= 1]

        # Range of returns (extend beyond individual asset returns for completeness)
        min_return = self.mu.min() - 0.02
        max_return = self.mu.max() + 0.02
    
        gamma_vals = np.logspace(-2, 3, num=num_points)
        
        objective = cp.Maximize(self.ret - self.gamma * self.risk)
        prob = cp.Problem(objective, constraints)        
        
        portfolios = []

        for i in range(num_points):
            w_rar, r_rar, v_rar = self.risk_adjusted_return_portfolio(gamma_value=gamma_vals[i])

            portfolios.append({
                'gamma': gamma_vals[i],
                'return': r_rar,
                'variance': v_rar,
                'std_dev': np.sqrt(v_rar),
                'weights': w_rar
                })
       
        print(f"Generated {len(portfolios)} efficient portfolios")

        return portfolios      


if __name__ == "__main__":
    # Load the CSV file back into a dataframe
    returns_df = pd.read_csv('./data/portfolio_returns.csv', index_col=0)

    # Initialize optimizer
    optimizer = CVXPYOptimizer(returns_df)

    # Global minimum variance portfolio
    w_mv, r_mv, v_mv = optimizer.minimum_variance_portfolio()
    # print(w_mv, r_mv, v_mv)

    w_rar, r_rar, v_rar = optimizer.risk_adjusted_return_portfolio(gamma_value=1.0)
    # print(w_mv, r_mv, v_mv)

    SAMPLES = 100
    # Generate and plot efficient frontier
    portfolios = optimizer.efficient_frontier_cvxpy(num_points=50)

    efficient_frontier_risk_data = [p['std_dev'] for p in portfolios]
    efficient_frontier_return_data = [p['return'] for p in portfolios]

    asset_risk_data = np.sqrt(np.diag(optimizer.sigma))
    asset_return_data = optimizer.mu[0]
    asset_labels = [col for col in returns_df.columns]

    point_risk_data = [np.sqrt(v_mv)]
    point_return_data = [r_mv]
    point_labels = ["Minimum variance portfolio", "Tangent portfolio"]

    plot_frontier(efficient_frontier_risk_data, efficient_frontier_return_data, asset_risk_data, asset_return_data, asset_labels, point_risk_data, point_return_data, point_labels, annualize_data=False)
