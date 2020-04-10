function [Ns,state_para] = cpbic_analysis(data)

% Determine the number of states and state parameters from input trajectory.

% data: raw trajectory (in the format of row).
% Ns: number of states.
% state_para: parameters of each state.

cp = findcp(data); % change point analysis (with a default type-I error rate of 0.05).
Yi = AHclusterN(data,cp); % agglomerative hierachical(AH) clustering.
%Yem = AHstateN(Yi); 
% calcluate parameters from AH clustering. Use this line instead of the next 
% line if parameters are to be calculated without using EM clustering. 
Yem = EMclusterN(Yi); % calcluate parameters from EM clustering.
bic = BICtestN(Yi,Yem); % BIC analysis from the clustering results. 
k = (bic == max(bic));  
Ns = Yem{1}(k).nos; 
state_para = get_state(Ns,k,Yem);

end

function state_para = get_state(Ns,k,Yem)

% Extract parameters of states.

% state_para: 
% 1st column: intensity levels of states.
% 2nd column: noise levels(standard deviation) of states.
% 3rd column: populations of states.
% the number of rows in state_para corresponds to the number of states,
% each row include the parameters from one state. 

state_para = zeros(Ns,3);
g = 1;
s = 1;
while s <= Ns
    if ~ismember(Yem{g}(k).intensity,state_para(:,1))
        state_para(s,1) = Yem{g}(k).intensity;
        state_para(s,2) = Yem{g}(k).sigma;
        state_para(s,3) = Yem{g}(k).pk;
        s = s + 1;
    end
    g = g + 1;
    if g > length(Yem)
        break;
    end
end       

end
