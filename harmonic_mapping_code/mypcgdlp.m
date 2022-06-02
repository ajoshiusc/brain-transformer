function [x] = mypcgdlp(A,b,x,Tol,Maxit,p)
%x=ones(size(A,1),1);
g = p*(A'*((A*x-b).^(p-1)));
d = -g;
gtg = g'*g;
newgtg=0;
for i=1:Maxit
%     if gtg<Tol 
%         break;
%     end
    alpha=lplsrch(A,x,b,d,1e-12,p);
    x = x+alpha*d;
    g = p*(A'*((A*x-b).^(p-1)));
     
    newgtg=g'*g;
    beta=newgtg/gtg;
    gtg=newgtg;
    
    d=-g+beta*d;
end
%disp(sprintf('Mypcg did %d iterations Tol=%e',i,g'*g));


