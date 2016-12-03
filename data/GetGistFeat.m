function GetGistFeat()
dataset='cifar10';
switch dataset
    case 'cifar10'
        ProcessCIFAR10();
    otherwise
        disp('cannot find the data.')
end
      
function ProcessCIFAR10()
addpath('gist\')
datapath='E:\DevProj\Datasets\CIFAR10\cifar-10-batches-mat\';
dictDataProc={'data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch'};
nr=32;
nc=32;
numPixel=nr*nc;
img=zeros(nr,nc,3);

paramGist.orientationsPerScale = [8 8 8 8];
paramGist.numberBlocks = 4;
paramGist.fc_prefilt = 4;
dimGist=512;
cifar10_gist=cell(1,length(dictDataProc));
for ii=1:length(dictDataProc)
    datafile=[datapath,dictDataProc{ii}];
    dataContent=load(datafile);
    numImg=size(dataContent.data,1);
    gistFeat=zeros(numImg,dimGist);
    for jj=1:numImg
        for kk=1:3
            img(:,:,kk)=reshape(dataContent.data(jj,(kk-1)*numPixel+1:kk*numPixel),[nr nc])';
        end
        [gistFeat(jj,:),paramGist]=ExtractGist(img,'',paramGist);
        if (mod(jj,100)==0)
            disp(['Processing ',dictDataProc{ii},', ', num2str(jj), ' processed']);
        end
    end
    cifar10_gist{ii}.gistFeat=single(gistFeat);
    cifar10_gist{ii}.labels=dataContent.labels;
    cifar10_gist{ii}.batch_label=dataContent.batch_label;
end
save('cifar10_gist.mat','cifar10_gist');
