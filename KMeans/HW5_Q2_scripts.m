clear
clc

%{Load data point for Problem 2, HW5
X = load("../digit/digit.txt");
Y = load("../digit/labels.txt");
%}

%{
%problem 2.5.1, 2.5.2
[KMeanLabels, centroids, numiterations] = HW5_Kmeans.Kmean(6, X);
sumofsquares = HW5_Kmeans.groupSumSquares(centroids, KMeanLabels, X, 6);
[p1, p2, p3] = HW5_Kmeans.paircounting(Y, KMeanLabels);
%pause;
%}

%%{
%problem 2.5.3, 2.5.4
HW5_Kmeans;
sos_10 = zeros(10,1);
p1_10 = zeros(10,1);
p2_10 = zeros(10,1);
p3_10 = zeros(10,1);

for k=1:10
    sumofsquares = 0; p1_=0; p2_=0; p3_=0;
    for j=1:10
        [KMeanLabels, centroids, numiterations] = HW5_Kmeans.Kmean(k, X);
        sumofsquares = sumofsquares+ HW5_Kmeans.groupSumSquares(centroids, KMeanLabels, X, k);
        [p1, p2, p3] = HW5_Kmeans.paircounting(Y, KMeanLabels);
        p1_ = p1_ + p1;
        p2_ = p2_ + p2;
        p3_ = p3_ + p3;
    end
    sos_10(k,:) = sumofsquares/10;
    p1_10(k,:) = p1/10;
    p2_10(k,:) = p2/10;
    p3_10(k,:) = p3/10;

end
x = 1:1:10;
plot (x, sos_10);
pause
plot(x, p1_10);
hold on;
plot(x, p2_10);
plot(x, p3_10);
hold off;

%pause;
%%}




