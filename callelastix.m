function [status,result] = callelastix(ffile,mfile,outdir,pfile,varargin)
% ffile: fixed volume file (with file path)
% mfile: moving volume file
% outdir: directory for results
% pfile: param file
% optional additional inputs:
%   '-t0',filename : initial trasnsform file
%   '-fMask',filename : fixed mask
%   '-mMask',filename : moving mask


if nargin>4
c = 1;
flag = cell(1); fname = cell(1);
for i = 1:2:(nargin-4)
    flag{c} = varargin(i);
    fname{c} = varargin(i+1);
    c = c+1;
end
end

% build elastix command
CMD=sprintf('elastix -f %s -m %s -out %s -p %s',...
    ffile,...
    mfile,...
    outdir,...
    pfile);

if not(nargin<5)
    % append additional flags from name-value pairs
    for i = 1:length(flag)
        CMD=sprintf([CMD ' %s %s'],flag{i}{1},fname{i}{1});
    end
end
    
[status,result]=system(CMD);

end

