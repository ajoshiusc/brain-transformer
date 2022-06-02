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

function P=para_curve_circle_segmt(Z,FixPts,PFixPts)

    %Take 1st fixed pt to 1st location
     [FixPts,II]=sort(FixPts);
     PFixPts=PFixPts(II);
     %Z_old_indx=[[1:FixPts(1)-1]';[FixPts(1):size(Z,1)]'];
    Z_old_indx=[[FixPts(1):size(Z,1)]';[1:FixPts(1)-1]'];
    Z=[Z(FixPts(1):end,:);Z(1:FixPts(1)-1,:)];
    FixPts=FixPts-FixPts(1)+1;
    iii=find(FixPts<=0); FixPts(iii)=FixPts(iii)+size(FixPts,1)+1;
    




    FixPts  = squeeze([FixPts ;size(Z,1)+1]);
    PFixPts = squeeze([PFixPts;PFixPts(1)]);
    Z=[Z;Z(1,:)];
    P=zeros(size(Z,1),1);
    for i=1:length(FixPts)-1
        %parameterize arc
        P(FixPts(i):FixPts(i+1))=para_arc(Z(FixPts(i):FixPts(i+1),:),PFixPts(i),PFixPts(i+1));                
    end
%    P(1)=0;
  %P(FixPts)=PFixPts;
      
    P(Z_old_indx)=P(1:end-1);
    P=P(1:end-1);
    


    %This function parameterizes an arc
function Parc=  para_arc(Z_arc,S1,S2)

    numPts=size(Z_arc,1);
    lgths   = sqrt(sum((Z_arc(2:end,:)-Z_arc(1:end-1,:)).^2,2));
    totlgth = sum(lgths);
    if S2<S1
        S2=S2+2*pi;
    end
    unitp   = (S2-S1)/totlgth;
    Parc    = zeros(numPts,1);
    Parc(1) = S1;
    
    
    for i=1:numPts-1
        Parc(i+1)= Parc(i)+ lgths(i)*unitp;        
    end
    
    
    



