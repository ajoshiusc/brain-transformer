function [x] = mypcg(A,b,Tol,Maxit,M,x,flags)
%function [x] = mypcg(A,b,Tol,Maxit,M)
if ~exist('flags','var')
    flags='';
end
%  flags=strrep(flags,'-','');
%  a=strfind(flags,'v');
if isempty(strfind(flags,'v'))    
     verbosity=2;
else
    a=strfind(flags,'v');
    verbosity=flags(a(1)+1);   verbosity= str2double(verbosity);
end


if verbosity>1
    if Maxit>1000
        Maxit=round(Maxit/10);
        NITER=10;
    else
        NITER=1;
    end
else
    NITER=1;
end

if (~exist('x','var')) | isempty(x)
    x=zeros(size(A,1),1);
end
r=b-A*x;Minv=1./M;
p=Minv.*r;
z=p;
rtz1=zeros(Maxit,1);
rtz=r'*z;

for kk=1:NITER
    for i=1:Maxit
        if rtz<Tol % no need to take abs(rtz) for this preconditioner
            break;
        end
        Ap=A*p;
        alpha=(rtz)/(p'*Ap);
        x=x+alpha.*p;
        r=r-alpha.*Ap;
        z=Minv.*r;
        newrtz=r'*z;
        beta=newrtz/rtz;
        rtz=newrtz;
        p=z+beta.*p;
        rtz1(i)=rtz;
    end
    if rtz<Tol % no need to take abs(rtz) for this preconditioner
        break;
    end
    if verbosity>1
        disp1(sprintf('PCG: %d/%d',kk,NITER),'PCG',flags);
    end
end
%disp(sprintf('Mypcg did %d iterations',i));

