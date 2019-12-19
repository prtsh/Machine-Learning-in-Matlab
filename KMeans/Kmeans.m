
classdef HW5_Kmeans
     properties (Constant)        
        dataDir = '../digit';
        maxIter = 20;
     end
     
     methods (Static)
         function [KMeanLabels, centroids, numiterations] = Kmean(k, X)
             %1. Select k starting centers that are points from your data set
             %centroids = X(1:k,:);
             centroids =  HW5_Kmeans.getRandomCentroid(X, k);
             old_labels = zeros(size(X,1),1);
             numiterations = 0;
             
             for i = 1:HW5_Kmeans.maxIter 
                 converged =  true;
                 %2. Assign each data point to the cluster associated with the nearest of the k center points.
                 KMeanLabels = HW5_Kmeans.recalssify(k, X, centroids);
                 
                 %3. Re-calculate the centers as the mean vector of each cluster from (2).
                 centroids = HW5_Kmeans.recenter(k, X, KMeanLabels);
                 
                 %4. Define convergence as no change in label assignment from one step
                 %to another or you have iterated 20 times
                 for labels=1:size(KMeanLabels,1)
                     if KMeanLabels(labels) ~= old_labels(labels)
                         converged = false;
                     end
                 end
                 
                 if converged == true
                     numiterations = i;
                     break;
                 end
                 old_labels = KMeanLabels;
             end
         end
         
         %@function: classify X with given centroids, and k
         %@return: new labels for each data points
         function new_labels = recalssify(k,X, centroids)
             new_labels = zeros(size(X,1),1);
             datapoints = size(X,1);
             for data = 1:datapoints
                 mincost = intmax;     
                 bestlabel = 1;
                 for label = 1:k
                     cost = sqrt(pdist2(X(data, :), centroids(label, :)));
                     if mincost > cost
                         mincost = cost;
                         bestlabel = label;
                     end
                 end
                 new_labels(data) = bestlabel;
            end
         end
         
         %@function: calcualte new centroid for X and k
         %@return: new k centroids
         function new_centroid = recenter(k,X, labels)
             new_centroid = zeros(k, size(X,2));
             for cluster = 1:k
                  X_cluster = X(labels == cluster, :);
                  sum_cluster = sum(X_cluster, 1);
                  new_centroid(cluster,:) = sum_cluster/size(X_cluster,1);
             end
         end
           
         %@function: get k random centroids from X
         %@return: return the new centroids
         function centroids = getRandomCentroid(X, k)
             randidx = randperm(size(X, 1));
             centroids = X(randidx(1:k),:);
         end
         
         %@function: calculate group sum of squares
         %@return: group sum of squares
         function sumofsquares = groupSumSquares(centroid, labels, X, k)
              sumofsquares = 0;
              for i = 1:k
                  X_cluster = X(labels == i, :);
                  sumofsquares = sum(pdist2(X_cluster, centroid(i, :)));
             end
         end
         
         %@function: caluclate paircounting p1, p2, p3
         %@return: p1, p2, p3
         function [p1, p2, p3] = paircounting(Ylabels, KMeanLabels)
              totalpoints = size(Ylabels);
              p1 = 0; p2 = 0; 
              sameclass = 0; diffclass = 0;
              for i = 1:totalpoints
                  for j = i+1:totalpoints
                      %consider all pairs of same-class data points, 
                      if Ylabels(j) == Ylabels(i)
                          %let p1 be the percentage of pairs of which both 
                          %data points were assigned to the same cluster.
                          if KMeanLabels(j) == KMeanLabels(i)
                              p1 = p1 + 1;
                          end
                          sameclass = sameclass + 1;
                      end
                      
                      %Consider all pairs of different-class data points, 
                      if Ylabels(j) ~= Ylabels(i)
                          %let p2 be the percentage of pairs of which two data points are
                          %assigned to different clusters.
                          if KMeanLabels(j) ~= KMeanLabels(i)
                              p2 = p2 + 1;
                          end
                          diffclass = diffclass + 1;
                      end
                  end
              end
              
              p1 = p1/sameclass;
              p2 = p2/diffclass;
              p3 = (p1+p2)/2;
         end
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%End of Methods %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     end
end