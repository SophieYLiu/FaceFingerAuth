clc;
clear;
f = [];
F = [];
FP = [];
fp = [];
for k = 1:27
    folder = strcat('Database\Sub',num2str(k));
    path_f = strcat(folder, '\*.jpg');
    path_fp = strcat(folder, '\*.bmp');
    src_f = dir(path_f);
    src_fp = dir(path_fp);
 
    for i = 1:length(src_f)
        filename_f = strcat(folder, '\', src_f(i).name);
        I = imread(filename_f);
        img = rgb2gray(I);
        img = imresize(img, [256, 256]);
        f(i, :) = extractHOGFeatures(img);
    end
    for j = 1:length(src_fp)
        filename_fp = strcat(folder, '\', src_fp(j).name);
        I = imread(filename_fp);
        IMG = imresize(I, [256, 256]);
        fp(j, :) = extractHOGFeatures(IMG);
    end
    F = [F;f];
    FP = [FP;fp];
end   

%%%%%% FACE %%%%%%%
a = 0;
b = 260;
trupos_f = 0;
er_f = 0;
for i = 1:27
    label = [zeros(a,1);ones(10,1);zeros(b,1)];
    Par = cvpartition(label, 'Holdout', 0.3);
    train = F(Par.training,:);
    test = F(Par.test,:);
    %train
    %mysvm = fitcsvm(train, label(Par.training));
    %pred = predict(mysvm, test);
    B = TreeBagger(20,train,label(Par.training), 'Method', 'classification');
    pred = str2double(predict(B, test));
    
    errRate_f = sum(label(Par.test)~=pred)/Par.TestSize;
    er_mat_f(i,:) = errRate_f;
    er_f = er_f + errRate_f; %sum and to average later
    conf_f = confusionmat(label(Par.test), pred);
    x = conf_f(2,:); %get all actual positives
    trupos_f = trupos_f + x(2)/(x(1)+x(2));
    %increment and decrement
    a = a + 10;
    b = b - 10;
end
avg_trupos_f = trupos_f/27;
ag_er_f = er_f/27;

%%%%% FINGERPRINT %%%%%%
a = 0;
b = 260;
trupos_fp = 0;
er_fp = 0;
for i = 1:27
    label = [zeros(a,1);ones(10,1);zeros(b,1)];
    Par = cvpartition(label, 'Holdout', 0.3);
    train = FP(Par.training,:);
    test = FP(Par.test,:);
    %train
    %mysvm = fitcsvm(train, label(Par.training));
    %pred = predict(mysvm, test);
    B = TreeBagger(20,train,label(Par.training), 'Method', 'classification');
    pred = str2double(predict(B, test));
    
    errRate_fp = sum(label(Par.test)~=pred)/Par.TestSize;
    er_mat_fp(i,:) = errRate_fp;
    er_fp = er_fp + errRate_fp; %sum and to average later
    conf_fp = confusionmat(label(Par.test), pred);
    x = conf_fp(2,:); %get all actual positives
    trupos_fp = trupos_fp + x(2)/(x(1)+x(2)); %sum and to average later
    %increment and decrement
    a = a + 10;
    b = b - 10;
end
avg_trupos_fp = trupos_fp/27;
ag_er_fp = er_fp/27;
