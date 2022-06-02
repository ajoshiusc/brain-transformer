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
% along with SVREG.  If not, see <http://www.gnu.org/licenses/>.

function [VertConn,C] = vertices_connectivity_fast(FV,VERBOSE);
% function [VertConn,C] = vertices_connectivity_fast(FV,VERBOSE);
%
% Computes vertices connectivity fast
% 
%
% Authors: Dimitrios Pantazis, Anand Joshi, November 2007 

rowno=[FV.faces(:,1);FV.faces(:,1);FV.faces(:,2);FV.faces(:,2);FV.faces(:,3);FV.faces(:,3)];
colno=[FV.faces(:,2);FV.faces(:,3);FV.faces(:,1);FV.faces(:,3);FV.faces(:,1);FV.faces(:,2)];
data=ones(size(rowno));
C=sparse(rowno,colno,data);
C=spones(C);%(C>0)=1;
[rows,cols,vals] = find(C);
d = find(diff([cols' 0]));

%if there are vertices with no connections at all
nVertices = size(FV.vertices,1);
if length(d)~=nVertices
    VertConn = vertices_connectivity(FV);
    return
end

%fast calculation of vertices connectivity
VertConn = cell(nVertices,1); % one empty cell per vertex
VertConn{1} = rows(1:d(1))';
for i = 2:nVertices
    VertConn{i} = rows((d(i-1)+1):d(i))';
end


