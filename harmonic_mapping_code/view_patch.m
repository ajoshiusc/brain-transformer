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

function h=view_patch(FV1)
%function view_patch(FV)
%FV: a tessellation to view
%
%copywrite Dimitrios Pantazis, PhD student, USC

h=figure;
%camlight
%lighting gouraud
axis equal
axis off   
axis vis3d
FV.vertices=FV1.vertices;FV.faces=FV1.faces;
nVertices = size(FV.vertices,1);
hpatch = patch(FV,'FaceColor','interp','EdgeColor','none','FaceVertexCData',ones(nVertices,3)*0.9,'faceAlpha',1);%,'BackFaceLighting','unlit'); %plot surface        
lighting gouraud
set(gcf,'color','white'); light

