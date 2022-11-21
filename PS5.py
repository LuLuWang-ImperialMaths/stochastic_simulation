import numpy as np
import matplotlib.pyplot as plt

# EX1
# using Monte Carlo estimation
sigma = 1
N = 10000
a = 0

def p(x):
    return 1/ np.sqrt(2 * np.pi * sigma**2) * np.exp(-x**2 / (2 * sigma**2))

samples = np.array([])
for i in range(N):
    x = np.random.normal(0, 1)
    if x >= 4:
        a += 1

acc_prob = a / N

print(acc_prob)
# P(X>4) is too small to be sampled given N=10000

# using Importance Sampling estimation
# criterion: the expectation of the (test*weight) should be minimized and\
# that the func: (test^2p^2/q) should be integrable

mu = 6
sigma = 1
N = 10000
samples = np.array([])
a = 0

def q(x):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) /  np.sqrt(2 * np.pi * sigma**2)

for i in range(N):
    x = np.random.normal(6, 1)
    if x >= 4:
        w = p(x) / q(x)
        a += 1
        samples = np.append(samples, w)

sample_sum = np.sum(samples)
prob = sample_sum / N
print(a)

# try using a Cauchy (>4) (with a heavier tail)
from scipy.stats import cauchy

N = 10000
samples = np.array([])
a = 0

def c(x):
    return 1 / (np.pi * (1 + x**2))

for i in range(N):
    x = cauchy.rvs(loc=0, scale=1, size=1)  # a num up as loc and scale up
    if x >= 4:
        w = p(x) / c(x)
        a += 1
        samples = np.append(samples, w*x)

sample_sum = np.sum(samples)
prob = sample_sum / N
print(a)


# Sol to Ex1
# On notes Remark 4.3

xx = np.linspace(4, 20, 100000)

def p(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)  # N(0, 1)

def q(x, mu, sigma):
    # general N(mu, sigma)
    return np.exp(-(x-mu)**2/(2*sigma **2))/(np.sqrt(2*np.pi)*sigma)

def w(x, mu, sigma):  # weight
    return p(x)/q(x, mu , sigma)

I = np.trapz(p(xx), xx)  # AUC beyond X>4, necessitating IS
print('Integral of p(x) from 4 to infinity: ', I)

N = 10000
x = np.random.normal(0, 1, N)  # samples from p
I_est_MC = (1/N) * np.sum(x > 4) # MC estimation
print('Monte Carlo estimate: ', I_est_MC)

mu = 6
sigma = 1
x_s = np.zeros(N)
weights = np.zeros(N)

for i in range(N):
    x_s[i] = np.random.normal(mu, sigma, 1)
    weights[i] = w(x_s[i], mu, sigma)

I_est_IS = (1/N) * np.sum(weights * (x_s > 4))
print('Importance sampling estimate: ', I_est_IS)

weights = np.array([1, 2, 3])
x = np.array([2, 1, 2])

print((x>1))  # numerical to logical value
print(np.sum(weights * (x>1)))  # True gets multiplied by 1


# Q3
def p(x):
    return np.exp(-x**2 / 2) / np.sqrt(2*np.pi)

def q(x):
    return np.exp(-x**2 / (2*4)) / np.sqrt(2*np.pi*4)

# MC estimation
N = 100000
x = np.random.normal(0, 1, N)
MC_est = np.sum(x) / N
print(MC_est)

# IS estimation
y = np.random.normal(0, 2, N)
weights = np.array([p(y[i]) / q(y[i]) for i in range(N)])
IS_est = np.sum(weights * y) / N
print(IS_est)


# Q4
logw = np.array([1000, 1001, 999, 1002, 950])
max_log = max(logw)
dem1 = np.sum(np.exp(logw))
w = np.array([])

for j in range(4):
    w = np.append(w, np.exp(logw[j]) / dem1)

# -max(log) enables each item to be between 0 and 1
dem2 = np.sum(np.exp(logw - max_log))
w = np.array([])

for j in range(4):
    w = np.append(w, np.exp(logw[j] - max_log) / dem2)
# Python doesn't give an overflow warning

print(w) # even closer as number close 0 scaling ratio larger
