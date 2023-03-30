% SpeakerRecognition_Group13
% Created by A. Alexander, EPFL
% Modified by J. Richiardi
% Modified by S. Yanushkevich
% Modified by Group 13: B. Kramer, C. Truong. J. Lansang

%define the number of Gaussian invariants - could be modified
No_of_Gaussians=10;
%Reading in the data 
%Use audioread from matlab 
disp('-------------------------------------------------------------------');
disp('                    Speaker recognition Demo');
disp('                            using GMM');
disp('-------------------------------------------------------------------');

%-----------reading in the training data----------------------------------
training_data1=audioread('01_train.wav');
training_data2=audioread('02_train.wav');
training_data3=audioread('03_train.wav');

training_data_brian=audioread('training_data_brian.mp3');
training_data_jacob=audioread('training_data_jacob.mp3');

%------------reading in the test data------------------------------------
testing_data1=audioread('01_test.wav');
testing_data2=audioread('02_test.wav');
testing_data3=audioread('03_test.wav');

testing_data_brian=audioread("testing_data_brian.mp3");
testing_data_jacob=audioread('testing_data_jacob.mp3');

disp('Completed reading training and testing data (Press any key to continue)');
pause;

Fs=8000;   %uncoment if you cannot obtain the feature number from audioread above

%-------------feature extraction------------------------------------------
training_features1=melcepst(training_data1,Fs);
training_features2=melcepst(training_data2,Fs);
training_features3=melcepst(training_data3,Fs);

training_features_brian=melcepst(training_data_brian,Fs);
training_features_jacob=melcepst(training_data_jacob,Fs);

disp('Completed feature extraction for the training data (Press any key to continue)');
pause;


testing_features1=melcepst(testing_data1,Fs);
testing_features2=melcepst(testing_data2,Fs);
testing_features3=melcepst(testing_data3,Fs);

testing_features_brian=melcepst(testing_data_brian,Fs);
testing_features_jacob=melcepst(testing_data_jacob,Fs);

disp('Completed feature extraction for the testing data (Press any key to continue)');
pause;

%-------------training the input data using GMM-------------------------
%training input data, and creating the models required
disp('Training models with the input data (Press any key to continue)');

[mu_train1,sigma_train1,c_train1]=gmm_estimate(training_features1',No_of_Gaussians);
disp('Completed Training Speaker 1 model (Press any key to continue)');
pause;

[mu_train2,sigma_train2,c_train2]=gmm_estimate(training_features2',No_of_Gaussians);
disp('Completed Training Speaker 2 model (Press any key to continue)');
pause;

[mu_train3,sigma_train3,c_train3]=gmm_estimate(training_features3',No_of_Gaussians);
disp('Completed Training Speaker 3 model (Press any key to continue)');
pause;

[mu_train_brian,sigma_train_brian,c_train_brian]=gmm_estimate(training_features_brian',No_of_Gaussians);
disp('Completed Training Speaker Brian model (Press any key to continue)');
pause;

[mu_train_jacob,sigma_train_jacob,c_train_jacob]=gmm_estimate(training_features_jacob',No_of_Gaussians);
disp('Completed Training Speaker Jacob model (Press any key to continue)');
pause;

disp('Completed Training ALL Models  (Press any key to continue)');

pause;
%-------------------------testing against the input data-------------- 

%testing against the first model
[lYM,lY]=lmultigauss(testing_features1', mu_train1,sigma_train1,c_train1);
A(1,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train1,sigma_train1,c_train1);
A(1,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train1,sigma_train1,c_train1);
A(1,3)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_brian', mu_train1,sigma_train1,c_train1);
A(1,4)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_jacob', mu_train1,sigma_train1,c_train1);
A(1,5)=mean(lY);

%testing against the second model
[lYM,lY]=lmultigauss(testing_features1', mu_train2,sigma_train2,c_train2);
A(2,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train2,sigma_train2,c_train2);
A(2,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train2,sigma_train2,c_train2);
A(2,3)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_brian', mu_train2,sigma_train2,c_train2);
A(2,4)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_jacob', mu_train2,sigma_train2,c_train2);
A(2,5)=mean(lY);

%testing against the third model
[lYM,lY]=lmultigauss(testing_features1', mu_train3,sigma_train3,c_train3);
A(3,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train3,sigma_train3,c_train3);
A(3,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train3,sigma_train3,c_train3);
A(3,3)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_brian', mu_train3,sigma_train3,c_train3);
A(3,4)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_jacob', mu_train3,sigma_train3,c_train3);
A(3,5)=mean(lY);

%testing against the fourth model
[lYM,lY]=lmultigauss(testing_features1', mu_train_brian, sigma_train_brian, c_train_brian);
A(4,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train_brian, sigma_train_brian, c_train_brian);
A(4,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train_brian, sigma_train_brian, c_train_brian);
A(4,3)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_brian', mu_train_brian, sigma_train_brian, c_train_brian);
A(4,4)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_jacob', mu_train_brian, sigma_train_brian, c_train_brian);
A(4,5)=mean(lY);


%testing against the fifth model
[lYM,lY]=lmultigauss(testing_features1', mu_train_jacob, sigma_train_jacob, c_train_jacob);
A(5,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train_jacob, sigma_train_jacob, c_train_jacob);
A(5,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train_jacob, sigma_train_jacob, c_train_jacob);
A(5,3)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_brian', mu_train_jacob, sigma_train_jacob, c_train_jacob);
A(5,4)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_jacob', mu_train_jacob, sigma_train_jacob, c_train_jacob);
A(5,5)=mean(lY);

disp('Results in the form of confusion matrix for comparison');
disp('Each column i represents the test recording of Speaker i');
disp('Each row i represents the training recording of Speaker i');
disp('The diagonal elements corresponding to the same speaker');
disp('-------------------------------------------------------------------');
A
disp('-------------------------------------------------------------------');
% confusion matrix in color
figure; imagesc(A); colorbar;

disp('Results completed (Press any key to continue)');
pause;

% testing models against a probe /////////////////////////////////////////
disp('-------------------------------------------------------------------');
disp('                    Testing models against a probe');
disp('-------------------------------------------------------------------');

probe_data=audioread('probe.mp3');
probe_features=melcepst(probe_data,Fs);

% training probe model
[mu_train_probe,sigma_train_probe,c_train_probe]=gmm_estimate(probe_features',No_of_Gaussians);
disp('Completed Training Speaker Probe model (Press any key to continue)');
pause;


%testing against a speaker that does not exist in the data
[lYM,lY]=lmultigauss(testing_features1', mu_train_probe, sigma_train_probe, c_train_probe);
A(1,1)=mean(lY);

[lYM,lY]=lmultigauss(testing_features2', mu_train_probe, sigma_train_probe, c_train_probe);
A(1,2)=mean(lY);

[lYM,lY]=lmultigauss(testing_features3', mu_train_probe, sigma_train_probe, c_train_probe);
A(1,3)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_brian', mu_train_probe, sigma_train_probe, c_train_probe);
A(1,4)=mean(lY);

[lYM,lY]=lmultigauss(testing_features_jacob', mu_train_probe, sigma_train_probe, c_train_probe);
A(1,5)=mean(lY);

disp('Results in the form of confusion matrix for comparison');
disp('Each column i represents the test recording of Speaker i');
disp('Each row i represents the training recording of Speaker i');
disp('The diagonal elements corresponding to the same speaker');
disp('-------------------------------------------------------------------');
A
disp('-------------------------------------------------------------------');
% confusion matrix in color
figure; imagesc(A); colorbar;

disp('Results completed using a probe (Press any key to continue)');
pause;

% evaluating error rates using confusion matrix /////////////////////////////////////////
disp('-------------------------------------------------------------------');
disp('          Evaluating error rates using confusion matrix');
disp('-------------------------------------------------------------------');