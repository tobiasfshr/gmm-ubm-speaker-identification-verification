# gmm-ubm-speaker-identification-verification
Implementation of a speaker identification and a speaker verification system based on Gaussian Mixture Models (GMM) in combination with and Universal Background Model (UBM) on the YOHO dataset in MATLAB. For detailed description and results see report.

Run main.m to execute all tasks of the exercise including:

-initialisation:
-->load all datasets
-->pre-processing (front-end of speaker identification)

-creating GMM, UBM and adapted GMM-UBM
-->create plain GMM models
-->plot number of GMM components against performance
-->create UBM
-->adapt UBM with MAP estimation and training data

-speaker identification
-->create confusion matrices for speaker identification
-->for GMM
-->for GMM-UBM

-speaker verification
-->set thresholds according to training data and ubm data
-->conduct impostor trials
-->plot FAR, FRR and confusion matrices

IMPORTANT
-make sure to include your YOHO path in parameters.m, as the YOHO data has to be processed again (i didn't include the processed data as the datasets are quite big)
-make sure to include the VOICEBOX toolbox in paths
-the script will take quite long to run completely!
