function disp1(ss,modn,flags)

if ~exist('flags','var')
    flags='';
end

if isempty(strfind(flags,'v'))
    verbosity=2;
else
    a=strfind(flags,'v');
    verbosity=flags(a(1)+1);
end
verbosity=str2double(verbosity);
if verbosity == 0
    return;
end

if ~exist('modn','var')
    modn='';
end

if isempty(strfind(flags,'m'))
    modn='';
end
v=fix(clock);
if isempty(strfind(flags,'m'))
    if isempty(strfind(flags,'t'))
        fprintf('%s\n',ss);
    else
        fprintf('%02d:%02d:%02d %s\n',v(4),v(5),v(6),ss);
    end
else
    if isempty(strfind(flags,'t'))
        fprintf('%s: %s\n',modn,ss);
    else
        fprintf('SVREG:%s: %02d:%02d:%02d %s\n',modn,v(4),v(5),v(6),ss);
    end
end

%v=load([subbasename,'.verbosity.txt']);
