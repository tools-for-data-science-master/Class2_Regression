%% load the data 
[~, ~, raw] = xlsread('CraterParameterStatistics7.xlsx','Main');
raw = raw(3:326,[5,7:8,10,12,14,17:19]);

data = reshape([raw{:}],size(raw));

Dc = data(:,1);
CBA = data(:,2);
CBAsd = data(:,3);
dM = data(:,4);
Pmare = data(:,5);
Hmare = data(:,6);
Tc = data(:,7);
Phic = data(:,8);
Rho0 = data(:,9);

clearvars data raw;


X = [Dc Tc Rho0 Phic Pmare Hmare];
Y = CBA;


%% Example 1: Multivariate linear regression

%b = regress(Y,[ones(length(X),1) X]);
mdl = fitlm(X,Y)
plotResiduals(mdl)

%% locate and remove outliers
outlier = mdl.Residuals.Raw > 50;
index = find(outlier)
mdl = fitlm(X,Y,'Exclude',index)
%plotResiduals(mdl)
%mdl.ObservationInfo(index,:)

%% Use stepwise regression to simplify the model

stepwiselm(X,Y,'PEnter',0.06)

% multivariate linear model with cross-validation 
% CVMdl = fitrlinear(X,Y,'KFold',5,'Learner','leastsquares','Lambda',0);
% mse = kfoldLoss(CVMdl,'mode','individual');
% dev = mean(mse); 

%% Correlation analysis

% normalize the data
Y1 = (Y-mean(Y))/std(Y);
X1 = (X-repmat(mean(X),length(X),1))./repmat(std(X),length(X),1);

% plot the covariance    
[R,Pvalue]=corrplot([Y1 X1],'varNames',{'CBA','Dc','Tc','\rho0','\phic','Pmare','Hmare'},'tail','both');

%% Example 2: Lasso Regression
format long
% test the PENALIZED codes

%%lasso :https://blogs.mathworks.com/loren/2011/11/29/subset-selection-and-regularization-part-2/
[B,Stats] = lasso(X,Y, 'CV', 5);

%lassoPlot(B, Stats, 'PlotType', 'CV')
%B(:,Stats.IndexMinMSE)

figure;
subplot(2,1,1)

errorbar(Stats.Lambda,Stats.MSE,Stats.SE,'k');
set(gca,'xscale','log')
axis tight
hold on;
plot([0 max(Stats.Lambda)],[0.0017 0.0017],'k')
ylabel 'MSE (mGal^2)'
subplot(2,1,2)
imagesc(Stats.Lambda,1:6,abs(B)./repmat(max(abs(B),[],2),1,size(B,2)));
set(gca,'ytick',1:5,'yticklabel',{'Dc','Tc','\rho0','\phic','Pmare','Hmare'},'Layer','top')

colormap jet
axis tight
xlabel 'Lambda'

%% relaxed LASSO 

model = glm_gaussian(Y,X);
%fit = penalized(model,@p_flash,'delta',0:0.1:1,'lambdamax',0.1,'standardize','true');
cv = cv_penalized(model,@p_flash,'lambdamax',1,'delta',0:0.1:1,'folds',5,'dev','deviance');


figure;
[yy,xx] = meshgrid(cv.delta,cv.lambda);
pcolor(xx,yy,cv.mse)
set(gca,'xscale','log');
%set(gca,'ColorScale','log')
axis tight
ylabel '\delta'
xlabel '\lambda'
colormap jet
colorbar
shading flat
%freezeColors
hold on 
%contour(xx,yy,cv.mse,[0.32 0.32],'w','linewidth',2)
%caxis([0.0015 0.0056])
title 'Cross-validated MSE'

figure
%imagesc(cv.lambda,cv.delta,cv.fit.nz');
pcolor(xx,yy,cv.p);
set(gca,'xscale','log');
cmap = jet(9);colormap(cmap)
%caxis([0.5 9.5])
colormap jet
ylabel '\delta'
xlabel '\lambda'
axis tight
colorbar
shading flat
hold on;
%contour(xx,yy,cv.mse,[0.32 0.32],'w','linewidth',2)
ylim([0 1])
title 'Number of nonzero coefficients'


%% subset selection
% Follow this reference: https://blogs.mathworks.com/loren/2011/11/21/subset-selection-and-regularization/#19
% create an index for the regression subsets
N = size(X,2);
index = dec2bin(1:2^N-1);
index = index == '1';
results = double(index);
results(:,N+1) = zeros(length(results),1);

for i = 1:length(index)
    foo = index(i,:);
    regf = @(XTRAIN, YTRAIN, XTEST)(XTEST*regress(YTRAIN,XTRAIN));
    results(i,N+1) = crossval('mse', X(:,foo), Y,'kfold', 5, 'predfun',regf);
end

index = sortrows(results, N+1);
logical(index(1,1:N))';
[beta,~,~,~,stat] = regress(Y, [X(:,logical(index(1,1:N))) ones(length(X),1)]);

figure;
subplot1(2,1)
subplot1(1)
plot(1:length(index),index(:,end)','k');
axis tight
hold on;
ylabel 'MSE (mGal^2)'
plot([0 length(index)],[0.0017 0.0017],'k');

subplot1(2)
imagesc(1:2^N-1,1:N,index(:,1:end-1)');
set(gca,'ytick',1:9,'yticklabel',{'Dc','Tc','Phic','Rho0','Age','Mare','Hfill','R','Type'},'Layer','top')
colormap jet
axis tight
xlabel 'Lambda'

%% Example 3:  PCA analysis
[wcoeff,score,latent,~,explained] = pca(X,'VariableWeights','variance','Rows','pairwise');
coefforth = inv(diag(std(X)))* wcoeff;
%biplot(coefforth(:,1:2),explained(:,1:2))

X2 = score;
corrplot(score)


