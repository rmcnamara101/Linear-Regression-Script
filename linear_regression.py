#rewritten linear regression script
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

class linear_regression():
    def __init__(self, x_data: list, y_data: list, err_data: list, xlabel: str, ylabel: str, title: str) -> None:
        
        sns.set(style="whitegrid")
        self.x_values = x_data
        self.y_values = y_data
        self.err = err_data
        self.x_label = xlabel
        self.y_label = ylabel
        self.title = title

    def ave(self, inputs, weight):
        """
        Calculates the average of any array
        """
        s = 0
        sum_weights = 0
        for i in range(0,len(inputs)-1):
            s = s + weight[i]*inputs[i]
            sum_weights = sum_weights + weight[i]
        ave = s / sum_weights
        return ave
    
    def dev(self, xvalues, xmean, weight):
        """
        Calculates the weighted deviation of the x values from the mean
        """
        deviation = 0
        for i in range(0,len(xvalues)-1):
            deviation = deviation + weight[i]*(xvalues[i]-xmean)**2
        return deviation
    
    def grad(self, xvalues,yvalues,xmean,deviation,weights):
        """
        Calculates the gradient of the line of best fit
        """
        grad = 0
        for i in range(0,len(xvalues)-1):
            grad = grad + weights[i]*(xvalues[i]-xmean)*yvalues[i]
        grad = grad / deviation
        return grad

    def weight_function(self, delta_yvalues):
        """
        Calculates the weight of each uncertainty
        """
        ones = np.ones(len(delta_yvalues))
        weights = np.zeros(len(delta_yvalues))
        weights = np.divide(ones,delta_yvalues)**2
        return weights
    
    def d(self, y_values, x_values, grad, y_intercept):
        """
        Calculates the d value where y = y - mx - c
        """
        d = np.zeros(len(x_values))
        for i in range(0,len(x_values)):
            d[i] = y_values[i] - grad*x_values[i] - y_intercept
        return d
    
    def delta_c(self, weights, xmean, dev,d,):
        """
        Calculates the uncertainty in the y-intercept
        """
        sum_weights = 0
        sum_weights_d = 0
        for i in range(0,len(weights)):
            sum_weights = sum_weights + weights[i]
            sum_weights_d = sum_weights_d + weights[i]*d[i]**2
        delta_c_squared = (( 1 / sum_weights + xmean**2 / dev)*(sum_weights_d / (len(weights) - 2)))**0.5
        return delta_c_squared
    
    def delta_m(self, dev,weights,d,):
        """
        Calculates the uncertainty in the gradient
        """
        sum = 0
        for i in range(0,len(weights)):
            sum = sum + weights[i]*d[i]**2
        delta_m = (sum / (dev * (len(weights)-2)))**0.5
        return delta_m
    
    def calculate_r_squared(self, weights, m , c):
        """
        Calculates the R-squared value for the line of best fit.
        """
        y_mean = self.ave(self.y_values, weights)
        y_predicted = [self.x_values[i] * m + c for i in range(len(self.x_values))]
        sum_squared_total = sum((self.y_values[i] - y_mean) ** 2 for i in range(len(self.y_values)))
        sum_squared_residual = sum((self.y_values[i] - y_predicted[i]) ** 2 for i in range(len(self.y_values)))
        r_squared = 1 - (sum_squared_residual / sum_squared_total)
        return r_squared
    
    def calculate_p_value(self, dev, weights, d, m):
        """
        Calculates the p-value for the slope coefficient in the linear regression model.
        """
        n = len(self.x_values)  # Number of data points
        df = n - 2  # Degrees of freedom (n - number of estimated parameters)

        # Calculate the standard error of the slope coefficient
        std_error_slope = self.delta_m(dev, weights, d)

        # Calculate the t-statistic for the slope coefficient
        t_statistic = m / std_error_slope

        # Calculate the two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

        return p_value

    def calculate(self):

        weights = self.weight_function(self.err)

        xmean = self.ave(self.x_values, weights)
        ymean = self.ave(self.y_values, weights)
        xmax = max(self.x_values)
        xmin = min(self.x_values)
        dev = self.dev(self.x_values, xmean, weights)
        m = self.grad(self.x_values, self.y_values, xmean, dev, weights)
        c = ymean - m * xmean
        d = self.d(self.y_values, self.x_values, m, c)
        err_c = self.delta_c(weights, xmean, dev, d)
        err_m = self.delta_m(dev, weights, d)
        r_squared = self.calculate_r_squared(weights, m, c)
        p = self.calculate_p_value(dev, weights, d, m)

        lr_x = np.linspace(xmin,xmax,50)
        lr_y = m*lr_x+c
        if c > 0:
            equation = f'y ~ {round(m,3)}x + {round(c,3)}'
        else:
            equation = f'y ~ {round(m,3)}x - {round(abs(c),3)}'

        print(f'The gradient is: {m} +/- {err_m}')
        print(f'The y-intercept is: {c} +/- {err_c}')
        print(f'R^2: {r_squared}')
        print(f'p = {p}')
    
        plt.figure(figsize=(10, 8))
        plt.plot(lr_x, lr_y, linestyle='dashed', color='blue', label=equation, linewidth=2)
        plt.fill_between(lr_x, lr_y - err_m, lr_y + err_m, color='lightblue', alpha=0.5, label='Gradient Error')
        plt.errorbar(self.x_values, self.y_values, yerr=self.err, fmt='o', markersize=8, elinewidth=2, capsize=5, capthick=2, color='red', alpha=0.8, label='Experimental Data')
        plt.xlabel(self.x_label, fontsize=14)
        plt.ylabel(self.y_label, fontsize=14)
        plt.title(self.title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        sns.despine()
        custom_palette = sns.color_palette("husl", len(self.x_values))
        sns.set_palette(custom_palette)
        plt.legend(loc="upper left", fontsize=12)
        plt.tight_layout()
        plt.savefig("linear_regression_plot.png", dpi=300)
        plt.show()