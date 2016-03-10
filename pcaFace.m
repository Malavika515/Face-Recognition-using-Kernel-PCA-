clc;
close all;
clear all;

%Step - 1 : Reading Images and Creating Dataset i.e S (mn * P)
S = [];
M = 16;

for i = 1:M 
    path = 'Training\';
    path = strcat(path, int2str(i), '.jpg');
    I = imread(path);
    I = rgb2gray(I);
   % figure, imshow(I);
    I = reshape(I, [200*180, 1]);
    S = [S I];
end

%Step - 2 : Finding Mean and Creating Zero Mean Dataset i.e Mean and ZMD
[rows, cols] = size(S);
Mean = mean(S, 2);
ZMD = ones(rows, cols);
for i = 1:M
   Y = double(S(:, i));
   ZMD(:, i) = Y - Mean;
end

%Step - 3 : Drawing each mean Image.
for i = 1:M
    MeanI = reshape(ZMD(:, i), [200, 180]);
    %figure, imshow(MeanI);
end

%Step - 4 : Computing Covariance Matrix.
% 4.x : Computing Kernel Matrix. i.e. K .. (M * M)
gamma = 1;
sq_dists = pdist(ZMD', 'euclidean');
sq_dists = sq_dists .* sq_dists;
mat_sq_dists = squareform(sq_dists);
K = exp(-gamma * mat_sq_dists);

% 4.y : Normalizing Kernel Matrix.
oneK = zeros(M, M);
oneK = oneK + 1/M;
K = K - (oneK * K) - (K * oneK) + ((oneK * K)* oneK);

%Step - 5 : Calculating Eigen Values and Eigen Vectors.
C = K;
[Evtr Eval] = eig(C);

%Step - 6 : Drawing Feature Vector.
Fvtr = [];
for i = 1:M
    Fvtr = [Fvtr Evtr(:, i)];
end

%Step - 7 : Alignment of each mean face to the feature vector i.e. Avtr
Avtr = Evtr' * ZMD';  %3 * 36000
Avtr = Avtr';

figure, 
%Step - 7.x : Drawing Eigen Faces.
for i = 1:M
   Z = reshape(Avtr(:,i), [200, 180]);
   subplot(4, 4, i), imshow(Z);
end


%Step - 8 : Generating Signature of Eigen Faces i.e. Eface.
Eface = [];
for i = 1:M
   Eface = [Eface (Avtr' * ZMD(:, i))];
end

%Testing ....Considering Every Image...
for k = 1:M
%Step - 1 : Reading an Image
testI = imread(strcat('Training\', int2str(k), '.jpg'));
testI2 = testI;
testI = rgb2gray(testI);
testI = reshape(testI, [200*180, 1]);

%Step - 2 : Normalize an Image
testI = (double(testI)) - Mean;

%Step - 3 : Projecting Mean Aligned face to eigen face.
OmegaF = Avtr' *  testI;
idx = -1;
minE = inf;
for i = 1:M
    A = OmegaF;
    B = Eface(:, i);
    B = B - A;
    B = B.*B;
    B = sum(B);
    B = sqrt(B);
    if(minE > B && B ~= 0) 
        minE = B;
        idx = i;
    end
end

%Step - 4 : Declaration of Image.
figure, 
subplot(1, 2, 1), imshow(testI2), title('Input Image');
subplot(1, 2, 2), imshow(strcat('Training\', int2str(idx), '.jpg')), title('Output Image');
end
