function bic = BICtestN(Yi,Yem)

% BIC test of AH/EM clustering results. 

% bic: BIC values for each number of states(from 1 to Nst). 
% Yi: output from AH clustering using AHclusterN.m.
% Yem: output from AHstateN.m or EMclusterN.m.
% N: total number of data points.
% Ns_max: maximum number of states.
% Nst: practical number of states to be tested.
% ll: log-likelihood for the whole trajectory. 
% llc: classification log-likelihood. 
% G: number of states. 
% Ncp: effective number of change point according to each model.

N = sum([Yi(:).t]);
Ns_max = length(Yi);
if Ns_max > 10
  Nst = 10;
else
  Nst = Ns_max;
end
bic = zeros(1,Nst);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BIC for 1 group 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for G = 1
    ll = 0;
    llc = 0;
    for i = 1:Ns_max
        for j = 1:Yi(i).t
        gaus = normpdf(Yi(i).tr(j),Yem{i}(G).intensity,Yem{i}(G).sigma);
        ll = ll + log(gaus) + log(Yem{i}(G).pk); % 
        end  
        llc = llc + log(Yem{i}(G).prob);
    end
    bic(G) = ll + llc - (3/2)*G*log(Ns_max);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BIC for more than 1 group 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for G = 2:Nst
    ll = 0;
    Ncp = 0;
    llc = 0;
    for i = 1:Ns_max
        for j = 1:Yi(i).t
        gaus = normpdf(Yi(i).tr(j),Yem{i}(G).intensity,Yem{i}(G).sigma);
        ll = ll + log(gaus) + log(Yem{i}(G).pk); 
        % log-likelihood for each data point in the whole trajectory given
        % the parameters including CP, number of states, in each model. 
        end
        llc = llc + log(Yem{i}(G).prob);
        % log-likelihood (classification) for assigning ith segment to the Gth state. 
    end
    
    for k = 2:Ns_max
        if Yem{k}(G).class ~= Yem{k-1}(G).class
            Ncp = Ncp + 1;          
        end
    end
    if Ncp > 0
        bic(G) = ll + llc - (3/2)*G*log(Ncp) - (Ncp/2)*log(N);
    end
end

bic(bic==0) = [];

end