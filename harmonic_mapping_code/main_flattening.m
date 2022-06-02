clc;clear all;close all;
restoredefaultpath;
s=readdfs('/Users/omarzamzam/Desktop/data/colin27_32kleft.dfs');
load('/Users/omarzamzam/Desktop/data/USCBrain_grayordinate_labels.mat');
labels(32493:end, :) = [];
s.labels = labels;
trilabel = median(labels(s.faces), 2);
s.faces((trilabel==0),:) = [];
s = myclean_patch_cc(s);
patch('faces',s.faces,'vertices',s.vertices,'facevertexcdata',s.labels,'facecolor','flat');
colormap prism
axis equal
[xmap,ymap]=map_hemi(s);
figure;
patch('faces',s.faces,'vertices',[xmap,ymap,0*ymap],'facevertexcdata',s.labels,'facecolor','flat');
colormap prism
axis equal;
xind = linspace(-1,1,256);
yind = linspace(-1,1,256);
[X,Y] = meshgrid(xind, yind);
L = griddata(xmap, ymap, double(s.labels), X, Y, 'nearest');
imagesc(L);
colormap prism
axis equal