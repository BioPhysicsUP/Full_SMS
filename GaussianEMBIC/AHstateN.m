function Yem = AHstateN(Yi)

% Calculate state parameters using the clustering result from AH clustering without using EM.

% Yem: the clustering schemes as well as state parameters.
% Yi: output from AH clustering using AHclusterN.m.

Ng_max = length(Yi); % maximum number of groups
Yem = cell(1,Ng_max); % Matrix that store EM results
Tk = zeros(1,Ng_max); % Total time duration of the k-th group
pk = zeros(1,Ng_max); % Probability of getting the k-th group
yk = zeros(1,Ng_max); % mean of the k-th group
stdk = zeros(1,Ng_max); % std of the k-th group
pki = zeros(Ng_max,Ng_max-1); % Probability of the i-th section belonging to the k-th group
T = sum([Yi(:).t]); % Total time duration of the traj

if Ng_max > 10
    Ngt = 10; % Practical number of grouping to be test;
else
    Ngt = Ng_max;
end

for Ng = 1:Ngt  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize variable values 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:Ng_max    
    % parameters of the i-th trajectory segment under the clustering
    % scheme(model) with Ng total number of states.
    Yem{i}(Ng).class = 0; % the state which it belongs to. 
    Yem{i}(Ng).intensity = 0; % intensity level.
    Yem{i}(Ng).sigma = 0; % noise level.
    Yem{i}(Ng).prob = 0; % probability of observing this state.
    Yem{i}(Ng).pk = 0;  % probability of observing this segment.
    Yem{i}(Ng).nos = 0; % number of states.
    for k = 1:Ng
    if k == Yi(i).group(Ng)
        pki(i,k) = 1;
    else
        pki(i,k) = 0;
    end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate parameters of each state 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Ng
    mmt = [];
    wgt = [];
    Tk(k) = sum(pki(:,k)'.*[Yi(:).t]);
    pk(k) = Tk(k)/T;
    yk(k) = sum(pki(:,k)'.*[Yi(:).intensity])/sum(pki(:,k)); 
      for i = 1:Ng_max
          mmt = [mmt pki(i,k)*((Yi(i).tr-Yi(i).intensity).^2)]; 
          wgt = [wgt pki(i,k)*ones(1,length(Yi(i).tr))];
      end
    stdk(k) = sqrt(sum(mmt)/sum(wgt));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assign parameters to each segment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:Ng_max   
    k_class = find(pki(i,:) == max(pki(i,:)));
    k_class = k_class(1);      
    for k = 1:Ng
          pki(i,k) = pk(k)*pki(i,k);
    end   
    k_norm = sum(pki(i,1:Ng));

    Yem{i}(Ng).class = k_class; 
    Yem{i}(Ng).intensity = yk(k_class);  
    Yem{i}(Ng).sigma = stdk(k_class); 
    Yem{i}(Ng).prob = k_norm; 
    Yem{i}(Ng).pk = pk(k_class);
end

Ns = [];
for i = 1:length(Yem)
    Ns(i) = Yem{i}(Ng).class;
end

Ns = unique(Ns);
for i = 1:length(Yem)
    Yem{i}(Ng).nos = length(Ns);
end

end

end


