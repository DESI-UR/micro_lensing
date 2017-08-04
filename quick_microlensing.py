
import sys
import matplotlib.pyplot as plt
# add dithering module to path and load it
sys.path.append('./py')
import peakFinder
import extract_spectrum

wlens, fluxes = extract_spectrum.get_random_spectrum("QSO_SOUTH")
#plt.plot(wlens, fluxes)
#plt.show()

bkg = peakFinder.snip(fluxes, 20, 3)
plt.clf()
plt.plot(wlens, fluxes, alpha=0.5)
plt.plot(wlens, bkg)
plt.show()

#plt.clf()
#plt.plot(wlens, fluxes-bkg)
#plt.show()

source = fluxes-bkg
ssize  = len(fluxes)
sigma  = .25
decon_iterations = 20
max_peak = 50
threshold = peakFinder.get_FDR_threshold(source)
peak_pos = peakFinder.search_peak(source, ssize, sigma,
                                  threshold, decon_iterations,
                                  max_peak)
print(wlens[peak_pos])

fig, axes = plt.subplots(1, 1, figsize=(13, 5))
axes.plot(wlens, source, alpha=0.2)
axes.scatter(wlens[peak_pos], source[peak_pos], color = 'r')
axes.set(title = 'Data')
axes.set_xlim(3500, 4000)
fig.tight_layout()
plt.show()
