Downloaded package: matplotlib；scipy；openpyxl；pandas；numpy；scikit-learn；tqdm；torch；xgboost.

Preprocessing folder
1_Extract a single cycle.py: 
Capture the start and end of a single-cycle signal and save the extracted information to “cycle_data.xlsx” in the _cycle_start_end_ Excel folder.
2_Quality_assessment.py: 
Calculate the PSQI of the signal and save it to “cycle_data_psqi.xlsx.”
3_Preliminary Signal Screening.py: 
Save signals with PSQI values greater than 0.6 to “Preliminary_PSQI.xlsx”.

Feature Extraction and Blood Pressure Prediction
1.The 8D_KF folder contains the 1_Feature_extraction folder and the 2_BP_prediction folder.
(1)1_Feature_extraction folder:
01_Calculate_Knowledge_features.py: 
By plotting and analyzing the PPG signal and its fourth-order derivative, we interactively selected eight key knowledge feature points in different dimensions and performed a secondary screening of the signals based on waveform morphology. The extracted eight-dimensional knowledge feature data is saved in “01_result.xlsx”, while the final valid signals, after initial screening and waveform feature re-screening, are stored in the “raw_PP.xlsx”.

02_Write_blood_pressured_cycle_position.py:
Merge the blood pressure information (such as systolic and diastolic pressure) from the “raw_PPG.xlsx” file into the knowledge feature file, ultimately generating a comprehensive data file named “8D_KF.xlsx”. This file contains both the eight-dimensional knowledge features and their corresponding blood pressure information.

(2)2_BP_prediction folder: 
The study jointly constructed six predictive models, establishing models for predicting systolic and diastolic blood pressure based on the integrated 8-dimensional knowledge features.

2.The 9D_DF folder contains the 1_Feature_extraction folder and the 2_BP_prediction folder.
(1)1_Feature_extraction folder:
1_Calculate_Gaussian_optimal_parameters.py：
The optimal parameters of the unconstrained Gaussian model were calculated and saved in “1.Gaussian_optimal_parameters.xlsx”, and all generated Gaussian fitting visualization results were stored in the “Gaussian_Fitting_Plot” folder.计算了无约束高斯模型的最优参数，将其保存于 
2_Calculate_Gaussian_features.py: 
Calculate the corresponding Gaussian features based on the obtained optimal Gaussian parameters and save the results to the “2_Gaussian_features.xlsx”.
3_Write_blood_pressure.py: 
Merge the blood pressure information in “raw_PPG.xlsx” with the unconstrained Gaussian feature file, and save the integrated data to “3_Gaussian_features_BP.xlsx”, thereby obtaining a complete dataset containing the unconstrained 9-dimensional features and blood pressure information. The content of this file is identical to “9D_DF.xlsx”.
(2)2_BP_prediction folder: 
The study jointly constructed six predictive models, establishing models for predicting systolic and diastolic blood pressure based on the integrated 9-dimensional data features.
3.The 8D_KF+9D_DF folder contains the 1_Feature_extraction folder, the 2_BP_prediction folder, and the gaussian_fitting_plots folder.
(1)1_Feature_extraction folder:
1_Gaussian_boundary_settings.py: 
Based on the previously extracted knowledge features, initial parameters and constraint boundaries were prepared for the subsequent Gaussian fitting process, and the relevant results were saved in “1_Guassian_boundary.xlsx”.
2_Calculate_Gaussian_optimal_parameters.py: 
Calculate the optimal parameters of the Gaussian model based on the constraint boundaries and save them to the file "2.Gaussian_optimal_parameters.xlsx". At the same time, save all generated Gaussian fitting visualization results to the "gaussian_fitting_plot" folder.
3_Calculate_Gaussian_features.py: 
Calculate the corresponding Gaussian features based on the obtained optimal Gaussian parameters and save the results to the “3_Gaussian_features.xlsx”.
4_Write_blood_pressure.py: 
Merge the blood pressure data from "8D_KF.xlsx" with the Gaussian feature file, and save the integrated data to "4_Gaussian_features_BP.xlsx." This file combines the 8-dimensional knowledge features with the 9-dimensional data features, and also includes blood pressure information. Its content is exactly the same as "8D_KF 9D_DF.xlsx."
(2)2_BP_prediction folder:
This study constructed a total of six predictive models. The models are based on a 17-dimensional feature set, which integrates 8-dimensional knowledge features and 9-dimensional data features, and are used to predict systolic and diastolic blood pressure, respectively.
4.The 9D_DKF directory contains 36 data interval folders, each of which has three subfolders: 1_Feature_extraction, 2_BP_prediction, and gaussian_fitting_plots.
Take the 2% folder as an example.
(1)1_Feature_extraction folder:
1_Gaussian_data_intervals.py: 
Calculate the 2% data interval boundaries for each knowledge feature and save the results to “1_Data_interval_boundaries.xlsx”.
2_Calculate_Gaussian_optimal_parameters.py: 
Calculate the optimal parameters of the Gaussian model based on the 2% data interval boundaries of the knowledge features, save them to the "2.Gaussian_optimal_parameters.xlsx", and simultaneously save all the generated Gaussian fitting visualization results to "gaussian_fitting_plot".
3_Calculate_Gaussian_features.py: 
Calculate the corresponding Gaussian features based on the obtained optimal Gaussian parameters and save the results to the “3_Gaussian_features.xlsx”.
4_Write_blood_pressure.py: 
Merge the blood pressure data in '8D_KF.xlsx' with the Gaussian feature file. The integrated data is saved to '4_Gaussian_features_BP.xlsx', containing Gaussian features constrained by the 2% data range of knowledge features and blood pressure information.
Each interval's corresponding folder follows the same modeling process.

This study selected six predictive models to effectively predict blood pressure based on Gaussian features generated from 35 data interval constraints derived from knowledge characteristics.
本研究选用六种预测模型，以从知识特征中推导出的35个数据区间约束所生成的高斯特征为基础，实现血压的有效预测。

baseline
1.Add_baseline_features.py: 
The Gaussian features (DKF) constrained by a 95% data interval based on knowledge characteristics are fused with baseline features to construct an enhanced feature set. Among them, the Gaussian features come from "4_Gaussian_features_BP.xlsx", and the baseline features come from "PPG-BP dataset.xlsx". The final fused feature set will be saved to "4_Gaussian_features_BP_with_baseline.xlsx".
2.Handling_of_sex.py: 
Convert gender characteristics into binary code, with males encoded as 1 and females encoded as 0.
3.DNN.py: 
Refining blood pressure prediction using a DNN model with DKF that includes baseline features.对加入基线特征的DKF用DNN模型进行血压预测
PPGs+BIFs:
Under the DNN framework, two types of blood pressure predictions are calculated for each sample: one based solely on DKF features, and the other combining DKF features with baseline features.
在DNN框架下，分别计算各样本的两种血压预测值：一种仅基于DKF特征，另一种则结合了DKF特征与基线特征。

