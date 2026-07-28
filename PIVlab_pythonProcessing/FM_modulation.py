"""Auto-generated .py twin of FM_modulation.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



import numpy as np
import matplotlib.pyplot as plt


t=np.linspace(0, 1200, 10000)
R=138e-3
k0=6
dphideg=2
flib=1.5

dphi=dphideg*np.pi/180 #rad  
wlib=2*np.pi*flib
k=k0*np.pi/R
l=2*R/k0

U0=dphi*R*flib*np.cos(wlib*t) #m/s
wint=k*U0 #rad/s

p=np.cos(wint*t) #* np.sin(wlib*t) #rad


plt.figure()
plt.plot(t, p)


dt = t[1] - t[0]
N = p.size

P = np.fft.rfft(p) / N
f = np.fft.rfftfreq(N, dt)

plt.figure()
plt.plot(f, np.abs(P))
plt.axvline(flib, color='r', ls='--', label=r'$f_{lib}$')
plt.axvline(flib / 2, color='g', ls='--', label=r'$f_{lib}/2$')
plt.xlim(0, 5 * flib)
plt.xlabel('frequency [Hz]')
plt.ylabel('|FFT(p)|')
plt.legend()
plt.show()
