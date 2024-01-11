import numpy as np

from scipy.special import factorial

import matplotlib.pyplot as plt

N_cutoff = 15
d = 100

def G(t):
    """
    Evolution matrix element G_d,d-1(t), where
        G(t)=expm(tR)
    for Erlang chain
    """
    temp = d-1. + d*np.arange(N_cutoff)

    TX, TY = np.meshgrid(t,temp)

    return np.sum(TX**TY / factorial(TY) * np.exp(-TX),axis=0)

def p(t):
    """
    Evolution of p_d-1(t) for initial state p_0 = 1, rest p_i = 0.
    """
    temp = d + d*np.arange(N_cutoff)

    TX, TY = np.meshgrid(t,temp)

    return np.sum(TX**TY / factorial(TY) * np.exp(-TX),axis=0)

def F(t,tau):
    """
    Regularized out of time order correlator <<I(t+tau) I(t)>> for Erlang Clock
    """
    if t<np.inf:
        return (G(tau)*p(t) - p(t+tau)*p(t))*d**2
    else:
        return G(tau)*d - 1
    
def S(omega):
    """
    Regularized power-spectrum
    """
    return d+d / ((1+1.j*omega)**d * (1 - (1+1.j*omega)**d))

times = np.linspace(0,50,600)
omegas = np.linspace(0.01,30 / d,1000)

plt.figure()
plt.plot(times,G(times),label=r"$G_{0,d-1}(t)$")
# plt.plot(times,p(times),label=r"$p_{d-1}(t)$")
plt.xlabel(r"time $t = [1]$")
plt.ylabel(r"probability")
plt.legend(loc="best")
plt.savefig("figures/probabilities.jpg",dpi=300)
plt.show()

plt.figure()
plt.plot(times,F(np.inf,times),label=r"$\lim_{t\rightarrow\infty}F(\tau,t)$")
plt.xlabel(r"time $\tau = [1]$")
plt.ylabel(r"probability")
plt.legend(loc="best")
plt.savefig("figures/autocorrelation.jpg",dpi=300)
plt.show()

plt.figure()
plt.vlines(np.arange(1,6)*np.pi / d,np.min(2*np.real(S(omegas))),np.max(2*np.real(S(omegas))),linestyles="dashed",label=r"$\omega = \frac{k\pi}{d}$",colors="gray")
plt.plot(omegas,2*np.real(S(omegas)),label=r"$S(\omega)$")
# plt.plot(omegas,np.imag(S(omegas)),label=r"$\mathfrak{Im}S(\omega)$")
# plt.plot(omegas,np.abs(S(omegas)),label=r"$|S(\omega)|$")
plt.xlabel(r"frequency $\omega = [1]$")
plt.ylabel(r"power spectrum")
plt.legend(loc="best")
plt.savefig("figures/powerspectrum.jpg",dpi=300)
plt.show()

