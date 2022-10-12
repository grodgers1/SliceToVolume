function [status,result] = calltransformix(infile,tparam,outdir)

% build transformix command
if strcmp(infile,'')
    % spatial jacobian
    CMD=sprintf('transformix -jac all -out %s -tp %s',...
            outdir,...
            tparam);
elseif strcmp(infile(end-2:end),'txt')
    % points file
    CMD=sprintf('transformix -def %s -out %s -tp %s',...
            infile,...
            outdir,...
            tparam);
elseif strcmp(infile(end-2:end),'mha')
    % volume warping
    CMD=sprintf('transformix -in %s -out %s -tp %s',...
            infile,...
            outdir,...
            tparam);
else
    error('File format not recognized, must be .txt (points), .mha (volume), or empty string (spatjac)')
end

[status,result]=system(CMD);

end

