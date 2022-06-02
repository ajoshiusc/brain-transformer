% Copyright 2010 Anand A. Joshi, David W. Shattuck and Richard M. Leahy 
% This file is part SVREG.
% 
% SVREG is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% SVREG is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Lesser General Public License for more details.
% 
% You should have received a copy of the GNU Lesser General Public License
% along with BSE.  If not, see <http://www.gnu.org/licenses/>.

function alpha = lplsrch(A,x,b,d,Tol,p)
alpha=0;
iterNo =1;
    Ad=A*d;
    Adsqr=Ad.*Ad;
Axxb=A*x-b;
while (iterNo < 100)
iterNo = iterNo + 1;
	x_plus_alphad=x+alpha*d;
    %Ax=A*x_plus_alphad;
	Axb=Axxb+alpha*Ad;
        Axbp_2=(Axb).^(p-2);
    Lprime_a=sum((Axbp_2.*Axb).*(Ad));
	Ldprime_a=(p-1)*sum(Axbp_2.*(Adsqr)); 
	updt= Lprime_a/Ldprime_a; 

if abs(updt)<Tol
    break;
end
       alpha = alpha - updt;

end
 
