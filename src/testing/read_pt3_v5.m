function [trace,delaytimes] = read_pt3_v4(timeres,pathname,filename1)

% MODIFIED to include binning of counts into user-specified bins (timeres) + output of only counts and lifetime data
% 
% 
% PicoHarp 300    File Access Demo in Matlab
% This script reads a PicoHarp T3 Mode data file (*.pt3)
% Works with file format version 2.0 only!

% Tested with Matlab 6
% Peter Kapusta, Michael Wahl, PicoQuant GmbH 2007, updated May 2007

% This is demo code. Use at your own risk. No warranties.

% Note that marker events have a lower time resolution and may therefore appear 
% in the file slightly out of order with respect to regular (photon) event records.
% This is by design. Markers are designed only for relatively coarse 
% synchronization requirements such as image scanning. 

% T3 Mode data are written to an output file [filename].out 
% We do not keep it in memory because of the huge amout of memory
% this would take in case of large files. Of course you can change this, 
% e.g. if your files are not too big. 
% Otherwise it is best process the data on the fly and keep only the results.


%clear all;
%clc;

% User-adjustable initialisation for intensity-shift (change-point) algorithm
%timeres = 0.01; % time resolution in seconds



fid=fopen([pathname filename1]);

Ident = char(fread(fid, 16, 'char'));
FormatVersion = deblank(char(fread(fid, 6, 'char')'));
CreatorName = char(fread(fid, 18, 'char'));
CreatorVersion = char(fread(fid, 12, 'char'));
FileTime = char(fread(fid, 18, 'char'));
CRLF = char(fread(fid, 2, 'char'));
CommentField = char(fread(fid, 256, 'char'));

Curves = fread(fid, 1, 'int32');
BitsPerRecord = fread(fid, 1, 'int32');
RoutingChannels = fread(fid, 1, 'int32');
NumberOfBoards = fread(fid, 1, 'int32');
ActiveCurve = fread(fid, 1, 'int32');
MeasurementMode = fread(fid, 1, 'int32');
SubMode = fread(fid, 1, 'int32');
RangeNo = fread(fid, 1, 'int32');
Offset = fread(fid, 1, 'int32');
AcquisitionTime = fread(fid, 1, 'int32');
StopAt = fread(fid, 1, 'int32');
StopOnOvfl = fread(fid, 1, 'int32');
Restart = fread(fid, 1, 'int32');
DispLinLog = fread(fid, 1, 'int32');
DispTimeFrom = fread(fid, 1, 'int32');
DispTimeTo = fread(fid, 1, 'int32');
DispCountFrom = fread(fid, 1, 'int32');
DispCountTo = fread(fid, 1, 'int32');

for i = 1:8
DispCurveMapTo(i) = fread(fid, 1, 'int32');
DispCurveShow(i) = fread(fid, 1, 'int32');
end;

for i = 1:3
ParamStart(i) = fread(fid, 1, 'float');
ParamStep(i) = fread(fid, 1, 'float');
ParamEnd(i) = fread(fid, 1, 'float');
end;

RepeatMode = fread(fid, 1, 'int32');
RepeatsPerCurve = fread(fid, 1, 'int32');
RepeatTime = fread(fid, 1, 'int32');
RepeatWait = fread(fid, 1, 'int32');
ScriptName = char(fread(fid, 20, 'char'));

HardwareIdent = char(fread(fid, 16, 'char'));
HardwareVersion = char(fread(fid, 8, 'char'));
HardwareSerial = fread(fid, 1, 'int32');
SyncDivider = fread(fid, 1, 'int32');
CFDZeroCross0 = fread(fid, 1, 'int32');
CFDLevel0 = fread(fid, 1, 'int32');
CFDZeroCross1 = fread(fid, 1, 'int32');
CFDLevel1 = fread(fid, 1, 'int32');
Resolution = fread(fid, 1, 'float');

RouterModelCode      = fread(fid, 1, 'int32');
RouterEnabled        = fread(fid, 1, 'int32');

% Router Ch1
RtChan1_InputType    = fread(fid, 1, 'int32');
RtChan1_InputLevel   = fread(fid, 1, 'int32');
RtChan1_InputEdge    = fread(fid, 1, 'int32');
RtChan1_CFDPresent   = fread(fid, 1, 'int32');
RtChan1_CFDLevel     = fread(fid, 1, 'int32');
RtChan1_CFDZeroCross = fread(fid, 1, 'int32');
% Router Ch2
RtChan2_InputType    = fread(fid, 1, 'int32');
RtChan2_InputLevel   = fread(fid, 1, 'int32');
RtChan2_InputEdge    = fread(fid, 1, 'int32');
RtChan2_CFDPresent   = fread(fid, 1, 'int32');
RtChan2_CFDLevel     = fread(fid, 1, 'int32');
RtChan2_CFDZeroCross = fread(fid, 1, 'int32');
% Router Ch3
RtChan3_InputType    = fread(fid, 1, 'int32');
RtChan3_InputLevel   = fread(fid, 1, 'int32');
RtChan3_InputEdge    = fread(fid, 1, 'int32');
RtChan3_CFDPresent   = fread(fid, 1, 'int32');
RtChan3_CFDLevel     = fread(fid, 1, 'int32');
RtChan3_CFDZeroCross = fread(fid, 1, 'int32');
% Router Ch4
RtChan4_InputType    = fread(fid, 1, 'int32');
RtChan4_InputLevel   = fread(fid, 1, 'int32');
RtChan4_InputEdge    = fread(fid, 1, 'int32');
RtChan4_CFDPresent   = fread(fid, 1, 'int32');
RtChan4_CFDLevel     = fread(fid, 1, 'int32');
RtChan4_CFDZeroCross = fread(fid, 1, 'int32');

 ExtDevices = fread(fid, 1, 'int32');
 Reserved1 = fread(fid, 1, 'int32');
 Reserved2 = fread(fid, 1, 'int32');
 CntRate0 = fread(fid, 1, 'int32');
 CntRate1 = fread(fid, 1, 'int32');
 StopAfter = fread(fid, 1, 'int32');
 StopReason = fread(fid, 1, 'int32');
 Records = fread(fid, 1, 'uint32');
 ImgHdrSize = fread(fid, 1, 'int32');
 ImgHdr = fread(fid, ImgHdrSize, 'int32');

ofltime = 0;
cnt_1=0; cnt_2=0; cnt_3=0; cnt_4=0; cnt_Ofl=0; cnt_M=0; cnt_Err=0; % just counters
WRAPAROUND=65536;

syncperiod = 1E9/CntRate0;   % in nanoseconds

intcnt = 1;  % counter for intensity bins
ints = 1:ceil(AcquisitionTime*0.001/timeres);
delaytimes = zeros(1,Records);

for i=1:Records

T3Record = fread(fid, 1, 'ubit32');     % all 32 bits:
  
%   +-------------------------------+  +-------------------------------+ 
%   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|
%   +-------------------------------+  +-------------------------------+  
  
nsync = bitand(T3Record,65535);       % the lowest 16 bits:
  
%   +-------------------------------+  +-------------------------------+ 
%   | | | | | | | | | | | | | | | | |  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|
%   +-------------------------------+  +-------------------------------+  
  
chan = bitand(bitshift(T3Record,-28),15);   % the upper 4 bits:

%   +-------------------------------+  +-------------------------------+ 
%   |x|x|x|x| | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |
%   +-------------------------------+  +-------------------------------+

dtime=0;

switch chan;
    
% if chan = 1,2,3 or 4, then these  bits contain the dtime:
%   +-------------------------------+  +-------------------------------+ 
%   | | | | |x|x|x|x|x|x|x|x|x|x|x|x|  | | | | | | | | | | | | | | | | |
%   +-------------------------------+  +-------------------------------+    
    
case 1, cnt_1=cnt_1+1; dtime = bitand(bitshift(T3Record,-16),4095); delaytimes(cnt_1)=dtime.*Resolution;    % regular count at Ch1, Rt_Ch1 when the router is enabled
case 2, cnt_2=cnt_2+1; dtime = bitand(bitshift(T3Record,-16),4095);   % regular count at Ch1, Rt_Ch2 when the router is enabled
case 3, cnt_3=cnt_3+1; dtime = bitand(bitshift(T3Record,-16),4095);   % regular count at Ch1, Rt_Ch3 when the router is enabled
case 4, cnt_4=cnt_4+1; dtime = bitand(bitshift(T3Record,-16),4095);   % regular count at Ch1, Rt_Ch4 when the router is enabled
    
case 15                                         % This means we have a special record
   markers = bitand(bitshift(T3Record,-16),15); % where these four bits are markers:
     
%   +-------------------------------+  +-------------------------------+ 
%   | | | | | | | | | | | | |x|x|x|x|  | | | | | | | | | | | | | | | | |
%   +-------------------------------+  +-------------------------------+
    
        if markers==0                           % then this is an overflow record
        ofltime = ofltime + WRAPAROUND;         % and we unwrap the numsync (=time tag) overflow
%        cnt_Ofl=cnt_Ofl+1;
        else                                    % if nonzero, then this is a true marker event
%        cnt_M=cnt_M+1;
        end;
   
    otherwise  fprintf(fpout,'Err '); cnt_Err=cnt_Err+1; % such events should not occur in T3 Mode with the current routers
     
 end;
     
 truensync = ofltime + nsync;
 truetime = truensync.*syncperiod + dtime.*Resolution;

 % bin into binsize (note that the last counts are not included if they don't fit into a time bin!)
 % instead of specifying bins with a counter vector, ints and delay times are immediately binned
 if i==1
     prevtruetime = 0;    % previous time point
 end

 if (prevtruetime*1E-9<intcnt*timeres) && (truetime*1E-9>=intcnt*timeres)  % time to bin
     ints(intcnt) = cnt_1;  %these are accumulated counts
     intcnt=intcnt+1;
 end
 prevtruetime = truetime;
 if intcnt==5119
     
 end
 
 
 

end;


if intcnt<=length(ints) % In this case the truetime was shorter than 60s
    for i=intcnt:length(ints)
        ints(i)=cnt_1;
    end
        
end


ints=diff([0 ints]);   % these are now the real binned counts.


fclose(fid);

delaytimes=delaytimes(1:cnt_1);
%ints = ints./timeres; % convert to cps to conform to Danielis's original code
alltimes=timeres:timeres:timeres*length(ints);
trace=[alltimes' ints']; % conform to data output of Danielis's original code
%outfile2 = [pathname filename(1:end-4)];
%dlmwrite(outfile2,trace,' '); 

%x=.01:.01:13.16;
%hist(delaytimes(delaytimes>0),x)