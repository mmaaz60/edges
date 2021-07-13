% This script runs edge detection on the ORE images and save the edge boxes
% with the same name as of image.

function save_bbox_for_ORE_data( image_dir )
% Define and create the output directory
output_dir = "./edge_boxes";
mkdir(output_dir);
% Load the model
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=8;
% Set up opts
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect
% Get all .jpg files from directory
filePattern = fullfile(image_dir, '*.jpg'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
tic;
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    if rem(k, 2) == 0
        fprintf(1, 'Now reading Image No. %d\n', k);
        toc;
        tic;
    end
    save_image_edges(output_dir, fullFileName, model, opts);
end

end


function status = save_image_edges( output_dir, image_path, model, varargin )
% get default parameters (unimportant parameters are undocumented)
dfs={'name','', 'alpha',.65, 'beta',.75, 'eta',1, 'minScore',.01, ...
  'maxBoxes',1e4, 'edgeMinMag',.1, 'edgeMergeThr',.5,'clusterMinMag',.5,...
  'maxAspectRatio',3, 'minBoxArea',1000, 'gamma',2, 'kappa',1.5 };
o=getPrmDflt(varargin,dfs,1);

% run detector possibly over multiple images and optionally save results
bbox=edgeBoxesImg(image_path,model,o);
% Save it as mat file
tmp = convertCharsToStrings(image_path);
tmp = tmp.split('.'); tmp = tmp(1);
tmp = tmp.split('/'); tmp = tmp(end);
mat_path = output_dir + "/" + tmp + ".mat";
save(mat_path, 'bbox');
status=1;

end

function bbox = edgeBoxesImg( I, model, o )
% Generate Edge Boxes object proposals in single image.
if(all(ischar(I))), I=imread(I); end
model.opts.nms=0; [E,O]=edgesDetect(I,model);
if(0), E=gradientMag(convTri(single(I),4)); E=E/max(E(:)); end
E=edgesNmsMex(E,O,2,0,1,model.opts.nThreads);
bbox=edgeBoxesMex(E,O,o.alpha,o.beta,o.eta,o.minScore,o.maxBoxes,...
  o.edgeMinMag,o.edgeMergeThr,o.clusterMinMag,...
  o.maxAspectRatio,o.minBoxArea,o.gamma,o.kappa);
[h,w,c] = size(I);
for i = 1 : length(bbox)
    bbox(i,3) = bbox(i,3) + bbox(i,1);
    bbox(i,4) = bbox(i,4) + bbox(i,2);
    bbox(i,:) = bbox(i,:) ./ [w, h, w, h, 1];
end
end