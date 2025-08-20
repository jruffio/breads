import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.stats import median_abs_deviation
from scipy.ndimage import generic_filter
import matplotlib
matplotlib.use('tkAgg')

filename = 'lsq_fit_hr8799_b_ch1_1.fits'
hdu = fits.open(filename)
d = hdu['DATA'].data
M = hdu['MODEL_MATRIX'].data
linparas = hdu['LINPARAS'].data
wave = hdu['WAVE'].data
n = hdu['NOISE'].data
rows = hdu['ROWS'].data

print(np.nansum(n**2))
res = d - np.dot(M, linparas)
print(np.sqrt(np.nansum((res[100:600]**2))/len(res[100:600])))

plt.figure(figsize=(10, 8))
plt.suptitle(f'Fit results on {filename}', fontsize=16)

# === Subplot 1: Flux and Model Components ===
plt.subplot(3, 1, 1)
plt.plot(d, label='Data')
plt.plot(np.dot(M, linparas) - np.dot(M[:, 0], linparas[0]), label='Model (no planet)')
plt.plot(d - np.dot(M, linparas), label='Residuals')
plt.plot(np.dot(M[:, 0], linparas[0]), label='planet component')
plt.plot(n, label='Noise')
plt.xlabel('x')
plt.ylabel('Flux')
plt.title('Flux and Model Fit')
plt.legend()
plt.grid(True)

# === Subplot 2: Wavelength ===
plt.subplot(3, 1, 2)
plt.plot(wave, color='tab:orange')
plt.xlabel('x')
plt.ylabel('Wavelength (micron)')
plt.title('Wavelength Distribution')
plt.grid(True)

# === Subplot 3: Row Numbers ===
plt.subplot(3, 1, 3)
plt.plot(rows, color='tab:green')
plt.xlabel('x')
plt.ylabel('Column Number')
plt.title('Spatial Column Mapping')
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

data = fits.getdata('/Users/abidot/Desktop/miri_data_4829_test/HD 218396/12A/stage2/jw04829001001_07101_00002_mirifushort_cal.fits')
noise = fits.open('/Users/abidot/Desktop/miri_data_4829_test/HD 218396/12A/stage2/jw04829001001_07101_00002_mirifushort_cal.fits')['ERR'].data
bp = fits.open('/Users/abidot/Desktop/miri_data_4829_test/HD 218396/utils_fm_80_nodes/1A/jw04829001001_07101_00001_mirifushort_cal_roughbadpix.fits')['OLD_BADPIX'].data
plt.imshow(bp)
plt.show()
plt.plot(data[:, 487]+bp[:, 487])
plt.show()

colid=487
mad_threshold = 5
col_err = data[:, colid]#noise[:, colid]
d = data[:, colid]

#col_err[np.isnan(col_err)] = 0
col_err = col_err/generic_filter(col_err, np.nanmedian, size=50)

from astropy.stats import sigma_clip

clipped_data = sigma_clip(col_err, sigma=3, maxiters=5)
d[clipped_data.mask] = np.nan

plt.plot(d)
plt.show()