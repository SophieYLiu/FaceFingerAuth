clc;
clear;
concat = [];
f = [];
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
        f(i, :) = horzcat(extractHOGFeatures(img), extractLBPFeatures(img));
        %f(i, :) = extractLBPFeatures(img);
    end
    for j = 1:length(src_fp)
        filename_fp = strcat(folder, '\', src_fp(j).name);
        I = imread(filename_fp);
        img = imresize(I, [256, 256]);
        fp(j, :) = horzcat(extractHOGFeatures(img), extractLBPFeatures(img));
        %fp(j, :) = extractLBPFeatures(I);
    end
    %brute force paring 100 for each person
    T = [];
    for i = 1:length(src_f)
        tmp = [];
        for j = 1:length(src_fp)
            tmp(j, :) = horzcat(f(i,:), fp(j,:));
        end
        T = vertcat(T, tmp);
    end;
    concat = [concat;T];
end   

%partition
a = 0;
b = 2600;
trupos = 0;
res_er = [];
for i = 1:27
    label = [zeros(a,1);ones(100,1);zeros(b,1)];
    Par = cvpartition(label, 'Holdout', 0.3);
    train = concat(Par.training,:);
    test = concat(Par.test,:);
    %train
    mysvm = fitcsvm(train, label(Par.training));
    pred = predict(mysvm, test);
    %B = TreeBagger(20,train,label(Par.training), 'Method', 'classification');
    %pred = str2double(predict(B, test));
    errRate = sum(label(Par.test)~=pred)/Par.TestSize;
    res_er(i,:) = errRate;
    conf = confusionmat(label(Par.test), pred);
    x = conf(2,:); %get all actual positives
    trupos = trupos + x(2)/(x(1)+x(2));
    %increment and decrement
    a = a + 100;
    b = b - 100;
end
res_trupos = trupos/27;