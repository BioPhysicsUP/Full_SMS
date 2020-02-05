function Yi = AHclusterN(traj,cp)

% Agglomerative hierachical clustering of intensity trajectory with Gaussian noise. 

% traj: raw trajectory.
% cp: change points, output from fincp.m.

cp_sec = [1 cp length(traj)];
Ng_max = length(cp_sec)-1; % maximum number of states.
err_reg = 1; 
% data points within std of CP detection that are not included in intensity
% calculation of each segment. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign intensity levels to Yi 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:Ng_max
    % remove data points within the std of CP detection.
    if i == 1
       Yi(i).tr = traj(cp_sec(i):cp_sec(i+1)-err_reg);
       g(i).tr = traj(cp_sec(i):cp_sec(i+1)-err_reg);
    else
       Yi(i).tr = traj(cp_sec(i)+err_reg:cp_sec(i+1)-err_reg);
       g(i).tr = traj(cp_sec(i)+err_reg:cp_sec(i+1)-err_reg);
    end
Yi(i).intensity = mean(Yi(i).tr); % intensity level of trajectory segments.
Yi(i).t = length(Yi(i).tr); % length of trajectory segments.
Yi(i).group = [ones(1,length(cp_sec)-2),i]; % the group to which the segments belong to.
Yi(i).std = std(Yi(i).tr); % noise level of trajectory segments.

g(i).t = Yi(i).t;
g(i).m = 1;
g(i).member(1) = i; 

end
Ng = length(Yi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Agglomerative hierachical clustering 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while Ng > 1
for j = 1:Ng
    for i = 1:g(j).m
        Yi(g(j).member(i)).group(Ng) = j; 
        % save the result to Yi
    end
end

[ii,jj] = delta(g,Ng); % find the merging pairs, ii and jj, ii<jj

for j = 1:g(jj).m
    g(ii).member(g(ii).m+j) = g(jj).member(j);
    % merge the jj-th group into the ii-th group
end

g(ii).tr = [g(ii).tr g(jj).tr];
g(ii).t = g(ii).t + g(jj).t;
g(ii).m = g(ii).m + g(jj).m;
g(jj) = [];

Ng = Ng - 1;

end

end

function [ii,jj] = delta(g,Ng)

% find the merging pairs

dta_min = realmax;
for i = 1:Ng
    std_i = std(g(i).tr);
    int_i = mean(g(i).tr);
    
    for j = (i+1):Ng
        std_j = std(g(j).tr);
        int_j = mean(g(j).tr); 
        tr_ij = [g(i).tr g(j).tr];
        std_ij = std(tr_ij);
        int_ij = mean(tr_ij);
           
        dta = -log(2*pi*std_i*std_j)+log(2*pi*(std_ij^2))...
            +((int_i-int_ij)^2+(int_j-int_ij)^2)/(2*std_ij^2); 
        % log-likelihood ratio of merging the i-th and j-th states.

        if dta < dta_min
            ii = min(i,j);
            jj = max(i,j);
            dta_min = dta;
        end
    end
end

end






