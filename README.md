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
通过绘制并分析PPG信号及其四阶导数图，我们以交互方式选取了八个维度的关键知识特征点，并依据波形形态对信号进行了二次筛选。其中，提取的八维知识特征数据保存在 01_result.xlsx 中；而经过初筛与波形特征复筛后的最终有效信号，则存储于 raw_PP.xlsx 文件中。
02_Write_blood_pressured_cycle_position.py:
Merge the blood pressure information (such as systolic and diastolic pressure) from the “raw_PPG.xlsx” file into the knowledge feature file, ultimately generating a comprehensive data file named “8D_KF.xlsx”. This file contains both the eight-dimensional knowledge features and their corresponding blood pressure information.
将 raw_PPG.xlsx 文件中的血压信息（如收缩压、舒张压）合并至知识特征文件，最终生成一个名为 8D_KF.xlsx 的综合数据文件。该文件同时包含了八维知识特征及其对应的血压信息。
(2)2_BP_prediction folder: 
The study jointly constructed six predictive models, establishing models for predicting systolic and diastolic blood pressure based on the integrated 8-dimensional knowledge features.
研究共构建了六种预测模型，基于整合后的8维知识特征，分别建立针对收缩压和舒张压的预测模型。
2.The 9D_DF folder contains the 1_Feature_extraction folder and the 2_BP_prediction folder.
(1)1_Feature_extraction folder:
1_Calculate_Gaussian_optimal_parameters.py：
The optimal parameters of the unconstrained Gaussian model were calculated and saved in “1.Gaussian_optimal_parameters.xlsx”, and all generated Gaussian fitting visualization results were stored in the “Gaussian_Fitting_Plot” folder.计算了无约束高斯模型的最优参数，将其保存于 1.Gaussian_optimal_parameters.xlsx 中，同时将所有生成的高斯拟合可视化结果存入 Gaussian_Fitting_Plot 文件夹。
2_Calculate_Gaussian_features.py: 
Calculate the corresponding Gaussian features based on the obtained optimal Gaussian parameters and save the results to the “2_Gaussian_features.xlsx”.
基于得到的高斯最优参数计算相应的高斯特征，并将结果保存至 2_Gaussian_features.xlsx 文件中。
3_Write_blood_pressure.py: 
Merge the blood pressure information in “raw_PPG.xlsx” with the unconstrained Gaussian feature file, and save the integrated data to “3_Gaussian_features_BP.xlsx”, thereby obtaining a complete dataset containing the unconstrained 9-dimensional features and blood pressure information. The content of this file is identical to “9D_DF.xlsx”.将 raw_PPG.xlsx 中的血压信息与无约束高斯特征文件融合，整合后的数据保存至 3_Gaussian_features_BP.xlsx，从而获得包含无约束9维特征与血压信息的完整数据集。该文件内容与 9D_DF.xlsx 完全一致。
(2)2_BP_prediction folder: 
The study jointly constructed six predictive models, establishing models for predicting systolic and diastolic blood pressure based on the integrated 9-dimensional data features.
研究共构建了六种预测模型，基于整合后的9维数据特征，分别建立针对收缩压和舒张压的预测模型。
3.The 8D_KF+9D_DF folder contains the 1_Feature_extraction folder, the 2_BP_prediction folder, and the gaussian_fitting_plots folder.
(1)1_Feature_extraction folder:
1_Gaussian_boundary_settings.py: 
Based on the previously extracted knowledge features, initial parameters and constraint boundaries were prepared for the subsequent Gaussian fitting process, and the relevant results were saved in “1_Guassian_boundary.xlsx”.
根据先前提取的知识特征，为后续高斯拟合过程准备了初始参数及约束边界，相关结果保存至“1_Guassian_boundary.xlsx”。
2_Calculate_Gaussian_optimal_parameters.py: 
Calculate the optimal parameters of the Gaussian model based on the constraint boundaries and save them to the file "2.Gaussian_optimal_parameters.xlsx". At the same time, save all generated Gaussian fitting visualization results to the "gaussian_fitting_plot" folder.根据约束边界计算出高斯模型的最优参数，保存至文件 “2.Gaussian_optimal_parameters.xlsx”，同时将所有生成的高斯拟合可视化结果保存至 “gaussian_fitting_plot” 文件夹中。
3_Calculate_Gaussian_features.py: 
Calculate the corresponding Gaussian features based on the obtained optimal Gaussian parameters and save the results to the “3_Gaussian_features.xlsx”.
基于得到的高斯最优参数计算相应的高斯特征，并将结果保存至 “3_Gaussian_features.xlsx “。
4_Write_blood_pressure.py: 
Merge the blood pressure data from "8D_KF.xlsx" with the Gaussian feature file, and save the integrated data to "4_Gaussian_features_BP.xlsx." This file combines the 8-dimensional knowledge features with the 9-dimensional data features, and also includes blood pressure information. Its content is exactly the same as "8D_KF 9D_DF.xlsx."
将 “8D_KF.xlsx” 中的血压数据与高斯特征文件进行融合，整合后的数据保存至 “4_Gaussian_features_BP.xlsx”。该文件拼接了 8 维知识特征与 9 维数据特征，同时包含血压信息，其内容与 “8D_KF+9D_DF.xlsx” 完全一致。
(2)2_BP_prediction folder:
This study constructed a total of six predictive models. The models are based on a 17-dimensional feature set, which integrates 8-dimensional knowledge features and 9-dimensional data features, and are used to predict systolic and diastolic blood pressure, respectively.
本研究共构建了六种预测模型。模型基于由8维知识特征与9维数据特征整合而成的17维特征集，分别针对收缩压和舒张压进行预测建模。
4.The 9D_DKF directory contains 36 data interval folders, each of which has three subfolders: 1_Feature_extraction, 2_BP_prediction, and gaussian_fitting_plots.
Take the 2% folder as an example.
(1)1_Feature_extraction folder:
1_Gaussian_data_intervals.py: 
Calculate the 2% data interval boundaries for each knowledge feature and save the results to “1_Data_interval_boundaries.xlsx”.
计算各知识特征的2%数据区间边界，并将结果保存至“1_Data_interval_boundaries.xlsx”。
2_Calculate_Gaussian_optimal_parameters.py: 
Calculate the optimal parameters of the Gaussian model based on the 2% data interval boundaries of the knowledge features, save them to the "2.Gaussian_optimal_parameters.xlsx", and simultaneously save all the generated Gaussian fitting visualization results to "gaussian_fitting_plot".
根据知识特征的2%数据区间边界计算出高斯模型的最优参数，保存至文件 “2.Gaussian_optimal_parameters.xlsx”，同时将所有生成的高斯拟合可视化结果保存至 “gaussian_fitting_plot”。
3_Calculate_Gaussian_features.py: 
Calculate the corresponding Gaussian features based on the obtained optimal Gaussian parameters and save the results to the “3_Gaussian_features.xlsx”.
基于得到的高斯最优参数计算相应的高斯特征，并将结果保存至 “3_Gaussian_features.xlsx “。
4_Write_blood_pressure.py: 
Merge the blood pressure data in '8D_KF.xlsx' with the Gaussian feature file. The integrated data is saved to '4_Gaussian_features_BP.xlsx', containing Gaussian features constrained by the 2% data range of knowledge features and blood pressure information.
将 “8D_KF.xlsx” 中的血压数据与高斯特征文件进行融合，整合后的数据保存至 “4_Gaussian_features_BP.xlsx”包含知识特征的2%数据区间约束的高斯特征和血压信息。
Each interval's corresponding folder follows the same modeling process.

all 
This study selected six predictive models to effectively predict blood pressure based on Gaussian features generated from 35 data interval constraints derived from knowledge characteristics.
本研究选用六种预测模型，以从知识特征中推导出的35个数据区间约束所生成的高斯特征为基础，实现血压的有效预测。

baseline
1.Add_baseline_features.py: 
The Gaussian features (DKF) constrained by a 95% data interval based on knowledge characteristics are fused with baseline features to construct an enhanced feature set. Among them, the Gaussian features come from "4_Gaussian_features_BP.xlsx", and the baseline features come from "PPG-BP dataset.xlsx". The final fused feature set will be saved to "4_Gaussian_features_BP_with_baseline.xlsx".基于知识特征的95%数据区间约束的高斯特征（DKF）与基线特征进行融合，构建了增强型特征集。其中，高斯特征来源于“4_Gaussian_features_BP.xlsx”，基线特征来源于“PPG-BP dataset.xlsx”，最终融合后的特征集将保存至“4_Gaussian_features_BP_with_baseline.xlsx”
2.Handling_of_sex.py: 
Convert gender characteristics into binary code, with males encoded as 1 and females encoded as 0.
将性别特征进行二值化编码转换，即男性编码为1，女性编码为0。
3.DNN.py: 
Refining blood pressure prediction using a DNN model with DKF that includes baseline features.对加入基线特征的DKF用DNN模型进行血压预测
PPGs+BIFs:
Under the DNN framework, two types of blood pressure predictions are calculated for each sample: one based solely on DKF features, and the other combining DKF features with baseline features.
在DNN框架下，分别计算各样本的两种血压预测值：一种仅基于DKF特征，另一种则结合了DKF特征与基线特征。

