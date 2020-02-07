import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


#Q1 (a)
def plot_gaussian_pdf():
    mean = 0; std = 1; variance = np.square(std)
    x = np.arange(-5,5,.01)
    f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))

    plt.xlabel('x')
    plt.ylabel('probability')
    plt.title('Gaussian Distribution Mean 0, Std 1')
    plt.plot(x,f)
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
    return mean, std, skew, kurtosis


if __name__== '__main__':
    np.random.seed(496)
    plot_gaussian_pdf()
    plot_gaussian_hist(0 ,0.1)
    data_estimators(0, 0.1)
