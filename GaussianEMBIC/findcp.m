function cp = findcp(data) 

% To find change points from the input trajectory. 

% cp: array of the index of idnetified change points. 
% data: raw trajectory.

int=cell(1);
int{1}(1).x=data;
int{1}(1).minIndex=1;
int{1}(1).active=1;
traj = int{1};

changePointFound=1;
alpha=0.05; % type-I error rate
index=2;
prevMinIndex=1;
llr=[];
cp=[];

while changePointFound % find change points recursively using a binary segmentation approach.
    for i=1:length(traj)
        if traj(i).active
            prevMinIndex=traj(i).minIndex;
            [llrOut,kOut]=CPcall(traj(i).x);% find CP in the current segment                        
            if llrOut>CriVal(length(traj(i).x),1-alpha)    
                changePointFound=1;                                                        
                llr(end+1)=llrOut;
                cp(end+1)=traj(i).minIndex+kOut-1;                                              
                traj(i).active=0;
                traj(index).x=traj(i).x(1:kOut);% new segment on the left of the newest CP             
                traj(index).active=1;
                traj(index).minIndex=prevMinIndex;               
                index=index+1;
                traj(index).x=traj(i).x(kOut+1:end);% new segment on the right of the newest CP               
                traj(index).active=1;             
                traj(index).minIndex=kOut+prevMinIndex; 
                index=index+1;               
            elseif i==length(traj)               
                changePointFound=0;
                break;              
            elseif traj(i+1).active
                prevMinIndex=traj(i+1).minIndex;
                [llrOut,kOut]=CPcall(traj(i+1).x);% find CP in the right next segment              
                if llrOut>CriVal(length(traj(i+1).x),1-alpha) %!
                llr(end+1)=llrOut;
                cp(end+1)=traj(i+1).minIndex+kOut-1;%!              
                traj(i+1).active=0;                                             
                traj(index).x=traj(i+1).x(1:kOut);% new segment on the left of the newest CP
                traj(index).active=1;
                traj(index).minIndex=prevMinIndex;  
                index=index+1;
                traj(index).x=traj(i+1).x(kOut+1:end);% new segment on the right of the newest CP               
                traj(index).active=1;
                traj(index).minIndex=kOut+prevMinIndex;  
                index=index+1;                                                
                end
            end
        end       
    end
end

cp=unique(cp);

low_lim = 3; % allowed minimum distance between change points
cp(diff(cp)<low_lim)=[];
cp(cp<low_lim)=[];
cp(cp>(length(data)-low_lim))=[];
% remove change points that are too close to the near change point, or the start/end of a trajectory. 

end

function [llrt_max,k_max] = CPcall(x)

% calculate the maximum of log-likelihood ratio.
% x: trajectory(segment).
% llrt_max: maximum of log-likelihood ratio.
% k_max: the index of data point that has the maximum of log-likelihood ratio.

N=length(x);
wvar=var(x);
llrt=arrayfun(@(k) lr(k,N,x,wvar),1:N,'uniformoutput',false);
llrt=cell2mat(llrt');
llrt(isinf(llrt))=0;
[llrt_max,k_max]=max(llrt);

end

function lrt = lr(k,N,x,wvar)

% calculate the log-likelihood ratio(lrt). 
% k: index of data point. 
% N: total number of data point of the trajectory(segment).
% wvar: variance of the whole trajectory(segment).

lvar=var(x(1:k));
rvar=var(x((k+1):N));
lrt=abs(sqrt(N*log(wvar)-k*log(lvar)-(N-k)*log(rvar)));
    
end

function [cv] = CriVal(N,alpha)

% calculate the critical value (cv) given the number of data point(N) and type-I
% error rate. 

yd=-log(-1/2*log(alpha));
x=log(N);
a=sqrt(2*log(x));
b=2*log(x)+log(log(x));
cv=(yd+b)/a;

end





 
        
        
