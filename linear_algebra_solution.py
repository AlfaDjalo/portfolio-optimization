import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from charts import plot_frontier

class MeanVarianceOptimizer:
    """
    Mean-Variance Optimization using analytic linear algebra solutions
    
    The optimization problem:
    min (1/2) * w^T * Sigma * w
    subject to: w^T * mu = target_return (if specified)
                w^T * 1 = 1 (weights sum to 1)
    
    Solution uses Lagrange multipliers and matrix algebra    
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

        self.mu = self.returns_df.mean().values
        self.Sigma = self.returns_df.cov().values
        self.n = len(self.mu)

        # Precompute inverse of covariance matrix
        try:
            self.Sigma_inv = np.linalg.inv(self.Sigma)
        except np.linalg.LinAlgError:
            print("Covariance matrix is singular, using pseudo-inverse")
            self.Sigma_inv = np.linalg.pinv(self.Sigma)

        # Precompute useful quantities for analytic solutions
        self.ones = np.ones(self.n)

        # A = 1^T * Sigma^(-1) * 1 (scalar)
        self.A = self.ones.T @ self.Sigma_inv @ self.ones

        # B = mu^T * Sigma^(-1) * 1 (scalar)
        self.B = self.mu.T @ self.Sigma_inv @ self.ones

        # C = mu^T * Sigma^(-1) * mu (scalar)
        self.C = self.mu.T @ self.Sigma_inv @ self.mu

        # Discriminant for feasibility
        self.D = self.A * self.C - self.B**2

        print(f"Portfolio initalized with {self.n} assets")
        print(f"A = {self.A:.6f}, B = {self.B:.6f}, C = {self.C:.6f}")
        print(f"Discriminant D = A*C - B**2 = {self.D:.6f}")

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

        # Analytic solution
        weights = (self.Sigma_inv @ self.ones) / self.A

        # Portfolio statistics
        expected_return = weights.T @ self.mu
        variance = weights.T @ self.Sigma @ weights

        print(f"Weights: {weights}")
        print(f"Expected return: {expected_return:.6f}")
        print(f"Variance: {variance:.6f}")
        print(f"Standard deviation: {np.sqrt(variance):.6f}")

        # Verify weight constraint
        print(f"Sum of weights: {np.sum(weights):.6f} (should be 1.0)")

        return weights, expected_return, variance
    
    def target_return_portfolio(self, target_return):
        """
        Minimum variance portfolio for a given target return (analytic solution)
        
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
        print(f"MINIMUM VARIANCE FOR TARGET RETURN = {target_return:.6f}")
        print("="*50)

        if self.D <= 0:
            print("Warning: Problem may be infeasible (D <= 0)")

        # Solve the system of equations for Lagrange multipliers
        # [A B] [lambda1]   [1]
        # [B C] [lambda2]   [target_return]

        # Analytic solution for multipliers
        lambda1 = (self.C - self.B * target_return) / self.D
        lambda2 = (self.A * target_return - self.B) / self.D

        # Portfolio weights
        weights = lambda1 * (self.Sigma_inv @ self.ones) + lambda2 * (self.Sigma_inv @ self.mu)

        # Portfolio statistics
        expected_return = weights.T @ self.mu
        variance = weights.T @ self.Sigma @ weights

        print(f"Lagrange multipliers: l1 = {lambda1:.6f}, l2 = {lambda2:.6f}")
        print(f"Weights: {weights}")
        print(f"Expected return: {expected_return:.6f}")
        print(f"Variance: {variance:.6f}")
        print(f"Standard deviation: {np.sqrt(variance):.6f}")

        # Verify constraints
        print(f"Sum of weights: {np.sum(weights):.6f} (should be 1.0)")
        print(f"Return constraint error: {abs(expected_return - target_return):.6f}")

        return weights, expected_return, variance


    def tangency_portfolio(self, risk_free_rate=0.0):
        """
        Tangency portfolio (maximum Sharpe ratio) - analytic solution
        
        For the tangency portfolio, we maximize Sharpe ratio:
        max (w^T * mu - rf) / sqrt(w^T * Sigma * w)
        s.t. w^T * 1 = 1
        
        Solution: w = (Sigma^(-1) * (mu - rf*1)) / (1^T * Sigma^(-1) * (mu - rf*1))
        
        Parameters:
        risk_free_rate: risk-free rate
        
        Returns:
        weights, expected_return, variance, sharpe_ratio
        """

        print(f"\n" + "="*50)
        print(f"TANGENCY PORTFOLIO (Max Sharpe) - rf = {risk_free_rate:.6f}")
        print("="*50)

        # Excess return vector
        excess_returns = self.mu - risk_free_rate

        # Analytic solution
        numerator = self.Sigma_inv @ excess_returns
        denominator = self.ones.T @ numerator

        if abs(denominator) < 1e-10:
            print("Warning: Denominator near zero, solution may be unstable")

        weights = numerator / denominator

        # Portfolio statistics
        expected_return = weights.T @ self.mu
        variance = weights.T @ self.Sigma @ weights
        sharpe_ratio = (expected_return - risk_free_rate) / np.sqrt(variance)

        print(f"Weights: {weights}")
        print(f"Expected return: {expected_return:.6f}")
        print(f"Variance: {variance:.6f}")
        print(f"Standard deviation: {np.sqrt(variance):.6f}")
        print(f"Sharpe ratio: {sharpe_ratio:.6f}")
        
        # Verify weight constraint
        print(f"Sum of weights: {np.sum(weights):.6f} (should be 1.0)")
        
        return weights, expected_return, variance, sharpe_ratio        


    def efficient_frontier_analytic(self, num_points=100):
        """
        Generate efficient frontier using analytic solutions
        
        For any point on the efficient frontier with return μₚ:
        Variance = (A*μₚ² - 2*B*μₚ + C) / D
        
        This gives us the entire frontier analytically!
        """
        print(f"\n" + "="*50)
        print("EFFICIENT FRONTIER - ANALYTIC SOLUTION")
        print("="*50)

        # Range of returns (extend beyond individual asset returns for completeness)
        min_return = self.mu.min() - 0.02
        max_return = self.mu.max() + 0.02
        target_returns = np.linspace(min_return, max_return, num_points)

        # Analytic formula for efficient frontier
        variances = []
        portfolios = []

        for mu_p in target_returns:
            # Variance formula: σ²(μₚ) = (A*μₚ² - 2*B*μₚ + C) / D
            if self.D > 0:
                variance = (self.A * mu_p**2 - 2 * self.B * mu_p + self.C) / self.D

                # Only include if variance is non-negative
                # if variance >= 0:
                variances.append(variance)

                # Get actual portfolio weights for this return level
                weights, actual_return, actual_var = self.target_return_portfolio(mu_p)

                portfolios.append({
                    'target_return': mu_p,
                    'return': actual_return,
                    'variance': actual_var,
                    'std_dev': np.sqrt(actual_var),
                    'weights': weights
                    })

        print(f"Generated {len(portfolios)} efficient portfolios")

        return portfolios      

    def plot_efficient_frontier(self, portfolios):
        """
        Plot the efficient frontier and individual assets
        """        

        if not portfolios:
            print("No portfolios to plot")
            return
        
        returns = [p['return'] for p in portfolios]
        std_devs = [p['std_dev'] for p in portfolios]

        plt.figure(figsize=(12, 8))

        # Plot efficient frontier
        plt.plot(std_devs, returns, 'b-', linewidth=2, label='Efficient Frontier')

        # Plot individual assets
        asset_returns = self.mu
        asset_stds = np.sqrt(np.diag(self.Sigma))
        plt.scatter(asset_stds, asset_returns, c='red', s=100, alpha=0.7,
                    label='Individual Assets', zorder=5)
        
        # Add asset labels
        for i, col in enumerate(self.returns_df.columns):
            plt.annotate(col, (asset_stds[i], asset_returns[i]),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=10, fontweight='bold')
            
        # Highlight special portfolios
        # Minimum variance portfolio
        w_mv, r_mv, v_mv = self.minimum_variance_portfolio()
        plt.scatter(np.sqrt(v_mv), r_mv, c='green', s=150, marker='s',
                    label='Min Variance', zorder=5)
        
        # Tangency portfolio
        w_tan, r_tan, v_tan, sr_tan = self.tangency_portfolio()
        plt.scatter(np.sqrt(v_tan), r_tan, c='orange', s=150, marker='^',
                    label='Tangency (Max Sharpe)', zorder=5)
        
        plt.xlabel('Standard Deviation (Risk)', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        plt.title('Mean-Variance Efficient Frontier\n(Analytic Solution)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()        
 

if __name__ == "__main__":
    # Load the CSV file back into a dataframe
    returns_df = pd.read_csv('./data/portfolio_returns.csv', index_col=0)

    # Initialize optimizer
    optimizer = MeanVarianceOptimizer(returns_df)

    # Global minimum variance portfolio
    w_mv, r_mv, v_mv = optimizer.minimum_variance_portfolio()
    print(w_mv, r_mv, v_mv)

    # Portfolio with target return
    target_ret = 0.20 / 252
    w_target, r_target, v_target = optimizer.target_return_portfolio(target_ret)
    print(w_target, r_target, v_target)

    # Tangency portfolio (max Sharpe)
    rf_rate = 0.02 / 252
    w_tan, r_tan, v_tan, sr_tan = optimizer.tangency_portfolio(rf_rate)
    print(w_tan, r_tan, v_tan, sr_tan)

    # Generate and plot efficient frontier
    portfolios = optimizer.efficient_frontier_analytic(num_points=50)
    # optimizer.plot_efficient_frontier(portfolios)

    efficient_frontier_risk_data = [p['std_dev'] for p in portfolios]
    efficient_frontier_return_data = [p['return'] for p in portfolios]

    asset_risk_data = np.sqrt(np.diag(optimizer.Sigma))
    asset_return_data = optimizer.mu
    asset_labels = [col for col in returns_df.columns]

    point_risk_data = [np.sqrt(v_mv), np.sqrt(v_tan)]
    point_return_data = [r_mv, r_tan]
    point_labels = ["Minimum variance portfolio", "Tangent portfolio"]

    plot_frontier(efficient_frontier_risk_data, efficient_frontier_return_data, asset_risk_data, asset_return_data, asset_labels, point_risk_data, point_return_data, point_labels, annualize_data=False)
    # plot_efficient_frontier(efficient_frontier_risk_data, efficient_frontier_return_data, asset_risk_data, asset_return_data, annualize_data=True)

