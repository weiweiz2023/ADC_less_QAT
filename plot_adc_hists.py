import numpy as np
import matplotlib.pyplot as plt


def plot_hists():
    # version 2 captures before the round
    files = [
        # "/home/weiweiz/Documents/WW_03/saved/hist_csvs/test_resnet20+mnist_raw_input.csv" ,
        # "/home/weiweiz/Documents/WW_03/saved/hist_csvs/test_resnet20+mnist_raw_output.csv",
    #     "/home/weiweiz/Documents/WW_03/saved/hist_csvs/test_resnet20+mnist_pre_input.csv",
    #     "/home/weiweiz/Documents/WW_03/saved/hist_csvs/test_resnet20+mnist_pre_output.csv",
    #     "/home/weiweiz/Documents/WW_03/saved/hist_csvs/test_resnet20+mnist_pre_gf_input.csv",
    #    "/home/weiweiz/Documents/WW_03/saved/hist_csvs/test_resnet20+mnist_pre_gf_output.csv",
        #  "/home/weiweiz/Documents/WW_03/saved/hist_csvs/test_resnet20+mnist_pre_cal_input.csv",
        #  "/home/weiweiz/Documents/WW_03/saved/hist_csvs/test_resnet20+mnist_pre_cal_output.csv",
         "/home/weiweiz/Documents/WW_03/saved/hist_csvs/test_resnet20+mnist_pre_cal_gf_input.csv",
         "/home/weiweiz/Documents/WW_03/saved/hist_csvs/test_resnet20+mnist_pre_cal_gf_output.csv",
        
    ]
    names = [
        #  "QAT_ADC-input", "QAT_ADC-Output",
            #  "QAF_ADC-input", " QAF_ADC-Output",
            # "QAF_gf_ADC-input", " QAF_gf_ADC-Output",
            #   "QAF_cal_ADC-input", " QAF_cal_ADC-Output",
              "QAF_cal_gf_ADC-input", " QAF_cal_gf_ADC-Output",
             
             ]
    n = len(files)
    fig, axs = plt.subplots(n, 1, figsize=(12, 5 * n))
    
    if n == 1:
        axs = [axs]

    for i, file in enumerate(files):
        print(f"Plotting {file}")
        try:
            array = np.genfromtxt(file, delimiter=",")
            
            # 使用较少的bins以避免标签过于密集
            counts, bins, patches = axs[i].hist(array, bins=100, alpha=0.7, edgecolor='black')
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # 在所有非零柱形上添加标签
            for j, (count, x) in enumerate(zip(counts, bin_centers)):
                if count > 0:
                    axs[i].text(x, count + max(counts) * 0.01, 
                               f'{int(count)}',
                               ha='center', va='bottom', 
                               fontsize=7, rotation=45)  # 旋转45度避免重叠
            
            axs[i].set_title(names[i] if i < len(names) else files[i])
            axs[i].set_xlabel("Value")
            
            axs[i].set_ylabel("Frequency")
            axs[i].set_xlim(0, 60)
            axs[i].set_ylim(0, 3e5)
            
            axs[i].grid(True, alpha=0.3)
            #axs[i].set_ylim(0, max(counts) * 1.2)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    plt.tight_layout()
    plt.show()
    fig.savefig('./saved/hist_all_labels.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # 使用智能标签版本（推荐）
    plot_hists()
