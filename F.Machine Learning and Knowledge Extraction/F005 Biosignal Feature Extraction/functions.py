# Import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import biosppy as bp


def plot_ecg(ecg, fs):
    
    print('Plot ECG')
    out = bp.signals.ecg.ecg(ecg, fs, show=False)
    t = out['ts']
    ecg_filtered = out['filtered']
    rpeaks = out['rpeaks']

    plt.plot(t, ecg_filtered, color='tab:blue', linewidth=1, label='ECG signal')
    plt.plot(t[rpeaks], ecg_filtered[rpeaks],'kx', label='R-peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.show()
    
def plot_templates(ecg, fs):
    print('Plot Templates')

    out = bp.signals.ecg.ecg(ecg, fs, show=False)
    
    plt.plot(out['templates_ts'].T, out['templates'].T, color='tab:blue', linewidth=1, alpha=0.08)
    plt.plot(out['templates_ts'].T, np.mean(out['templates'],0).T, color='black', linestyle='--', linewidth=1, label='Mean')
    plt.legend()
    plt.show()

def compute_rri(ecg, fs):
    print('Compute RRIs')

    out = bp.signals.ecg.ecg(ecg, fs, show=False)

    rpeaks = out['rpeaks']/fs*1000 # convert sample indexes to ms
    rri = np.diff(rpeaks)
    
    # Compute mean and sd
    rr_mean = np.mean(rri)
    rr_sd = np.std(rri)
    
    print(f'Mean = {rr_mean}, SD = {rr_sd}')
    
    # Plot RRIs
    plt.figure(figsize=(10,3))
    plt.plot(rri, lw=1, color='k', marker='v', markerfacecolor='orange')
    plt.ylabel('RR-intervals (ms)')
    plt.show()
    
    # Plot histogram
    bins = np.arange(np.min(rri), np.max(rri)+10, 10)
    
    plt.hist(rri, bins, facecolor='skyblue', edgecolor='black')
    plt.title('RR Distribution')
    plt.xlabel('RR Interval (ms)')
    plt.ylabel('Count')
    plt.show()
    
def compute_hr(ecg, fs):
    
    out = bp.signals.ecg.ecg(ecg, fs, show=False)

    # Compute heart rate
    ts = out['rpeaks'][1:]/fs
    hr = fs * (60.0 / np.diff(out['rpeaks']))
    
    # Plot
    print('Plot HR')
    plt.figure(figsize=(10,3))
    plt.plot(ts, hr, lw=1, color='k', marker='v', markerfacecolor='orange')
    plt.ylabel('Heart Rate (bpm)')
    plt.xlabel('Time (s)')
    plt.show()
    
def compute_rmssd(ecg, fs):
    
    out = bp.signals.ecg.ecg(ecg, fs, show=False)

    rpeaks = out['rpeaks']/fs*1000 # ms
    rr_int = np.diff(rpeaks)
    rr_diff = np.diff(rr_int)
    
    rmssd = np.sqrt(np.mean(rr_diff**2))
    print(f'RMSSD = {rmssd} ms')
    
def plot_poincare(ecg, fs):
    print('Plot Poincaré')
    
    out = bp.signals.ecg.ecg(ecg, fs, show=False)

    rpeaks = out['rpeaks']/fs*1000 # convert sample indexes to ms
    rri = np.diff(rpeaks)
    
    x = rri[:-1]
    y = rri[1:]
    rr_mean = np.mean(rri)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title('Poincaré Plot', fontsize=14)
    ax.set_xlabel('$RR_i$ (ms)', fontsize=12)
    ax.set_ylabel('$RR_{i+1}$ (ms)', fontsize=12)
    
    # Plot Poincaré data points
    ax.scatter(x, y, marker='.', color='blue', alpha=0.5,s=100)
    ax.set_xlim([np.min(rri) - 50, np.max(rri) + 50])
    ax.set_ylim([np.min(rri) - 50, np.max(rri) + 50])
    ax.set_aspect(1./ax.get_data_ratio())
    
    
    # Draw RRi+1=RRi
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()])]  # max of both axes
    
    ax.plot(lims, lims, color='grey', linestyle='--')
    
    
    # Draw ellipse
    sd1 = np.std(np.subtract(x, y) / np.sqrt(2))
    sd2 = np.std(np.add(x, y) / np.sqrt(2))
    area = np.pi * sd1 * sd2
    
    
    ellipse_ = mpl.patches.Ellipse((rr_mean, rr_mean), sd1 * 2, sd2 * 2, angle=-45, edgecolor='black', facecolor="None")
    ax.add_artist(ellipse_)
    
    ax.grid()
    
    print(f'SD1 = {sd1} \nSD2 = {sd2} \nArea = {area} \nSD1/SD2 = {sd1/sd2}')

