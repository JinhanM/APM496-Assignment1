import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats


#Q1 (a)
def plot_gaussian_pdf():
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.xlabel('x')
    plt.ylabel('probability')
    plt.title('Gaussian Distribution Mean 0, Std 1')
    plt.savefig('/Users/jinhanmei/Desktop/Gaussian.png')
    plt.show()

#Q2(a)
def plot_gaussian_hist(mean, sigma):
    sample_data = np.random.normal(mean, sigma, 1000)
    count, bins, ignored = plt.hist(sample_data, 25, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mean)**2 /\
    (2 * sigma**2) ),linewidth=1, color='r')

    plt.xlabel('x')
    plt.ylabel('count')
    plt.title('Gaussian Distribution Mean 0, Std 0.1')
    plt.savefig('/Users/jinhanmei/Desktop/Gaussian_hist.png')

    plt.show()

#Q2.2
def data_estimators(mean, sigma):
    sample_data = np.random.normal(mean, sigma, 1000)
    mean = np.mean(sample_data)
    std = np.std(sample_data)
    skew = stats.skew(sample_data)
    kurtosis = stats.kurtosis(sample_data)
    print( mean, std, skew, kurtosis)


if __name__== '__main__':
    np.random.seed(496)
    plot_gaussian_pdf()
    plot_gaussian_hist(0 ,1)
    data_estimators(0, 1)
