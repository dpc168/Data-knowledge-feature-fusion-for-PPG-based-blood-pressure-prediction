import os
from tools_plus import *
from scipy.signal import savgol_filter
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

book1 = openpyxl.load_workbook('_cycle_start_end_Excel/cycle_data.xlsx')

sheet1 = book1['Sheet1']

sheet1_max_row = sheet1.max_row
print(sheet1_max_row)

PSQI_list = list()

for row in range(2, sheet1_max_row + 1):
    name = sheet1.cell(row, 1).value

    start = sheet1.cell(row, 2).value
    end = sheet1.cell(row, 3).value

    file_name = name + '.txt'

    path = "../dataset/" + file_name

    # 1. Read data
    ppg = Read_raw_ppg_signal(path)

    # 2. Extract single cycle
    ppg = ppg[start:end]

    # 3. Normalize
    ppg = min_max_normalization(ppg)

    # 4. Filter
    filter_ppg = savgol_filter(ppg, 101, 2)

    # 5. Calculate PSQI
    PSQI = calculate_perfusion_sqi(ppg, filter_ppg)

    PSQI_list.append(PSQI)

# Normalize PSQI_list
PSQI_list = min_max_normalization(PSQI_list)

flag = 0
for i in range(2, sheet1_max_row + 1):
    if PSQI_list[i - 2] >= 0.6:
        flag = flag + 1

    sheet1.cell(i, 4).value = PSQI_list[i - 2]

print(flag)

book1.save('cycle_data_psqi.xlsx')
book1.close()