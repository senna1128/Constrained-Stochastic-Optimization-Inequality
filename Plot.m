%% This fucntion draws plot for simulation
clear all; close all

A = dir(fullfile('./Figure/*'));
if ~isempty(A)
    for k = 3:length(A)
        delete(strcat('./Figure/', A(k).name))
    end
    fprintf('remove result file. Done!\n')
else
    fprintf('No result file.\n')
end


load('./Solution/AdapNewton.mat')
load('./Solution/AdapGD.mat')
load('./Solution/NonAdap.mat')


%% global color
col = horzcat(hsv(3),ones(3,1)*0.5)';
col2 = col(:,2:3); 


%% Plot KKT residual with constant 1
for step = 1:6 
    for sigma = 1:5
        AR{sigma} = Res{sigma,1}.KKT;
        BR{sigma} = ResG{sigma,1}.KKT;
        if length(ResN{step,sigma}.KKT)>0
            CR{sigma} = ResN{step,sigma}.KKT;
        else
            CR{sigma} = [NaN];
        end 
    end
    data=vertcat(AR,BR,CR)';
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    Mlab={'AdapNewton','AdapGD','NonAdapSQP'};
    figure(step)
    multiple_boxplot(data,xlab,Mlab,col,1,0)
%    filename = ['./Figure/KKTStep' num2str(step) '.png'];
%    print('-dpng', filename)
end


%% Plot KKT residual with varying constant
for cons = 1:4 
    for sigma = 1:5
        AR{sigma} = Res{sigma,cons}.KKT;
        BR{sigma} = ResG{sigma,cons}.KKT;
    end
    data=vertcat(AR,BR)';
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    Mlab={'AdapNewton','AdapGD'};
    figure(10+cons)
    multiple_boxplot(data,xlab,Mlab,col2,1.5,0)
%    filename = ['./Figure/KKTCons' num2str(cons) '.png'];
%    print('-dpng', filename)
end


%% Plot Sample size
for cons = 1:4
    for sigma = 1:5
        AR{sigma} = Res{sigma,cons}.CountG;
        BR{sigma} = ResG{sigma,cons}.CountG;
    end
    data=vertcat(AR,BR)';
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    Mlab={'AdapNewton','AdapGD'};
    figure(100+cons)
    multiple_boxplot(data,xlab,Mlab,col2,2,0)    
%    filename = ['./Figure/GSampleCons' num2str(cons) '.png'];
%    print('-dpng', filename)
end


for cons = 1:4
    for sigma = 1:5
        AR{sigma} = Res{sigma,cons}.CountF;
        BR{sigma} = ResG{sigma,cons}.CountF;
    end
    data=vertcat(AR,BR)';
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    Mlab={'AdapNewton','AdapGD'};
    figure(200+cons)
    multiple_boxplot(data,xlab,Mlab,col2,3,0)    
%    filename = ['./Figure/FSampleCons' num2str(cons) '.png'];
%    print('-dpng', filename)
end

%% Plot time
for cons = 1:4
    for sigma = 1:5
        AR{sigma} = Res{sigma,cons}.Time;
        BR{sigma} = ResG{sigma,cons}.Time;
    end
    data=vertcat(AR,BR)';
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    Mlab={'AdapNewton','AdapGD'};
    figure(300+cons)
    multiple_boxplot(data,xlab,Mlab,col2,4,0)    
%    filename = ['./Figure/Time' num2str(cons) '.png'];
%    print('-dpng', filename)
end


%% Plot Stepsize
cmap = jet(5);
for sigma = 1:5
    figure(1000)
    subplot(5,1,sigma)
    ProbId = Res{sigma,1}.ProbId;
    for ii = 1:5
        plot(Res{sigma,1}.Alpha{ProbId(ii)}, 'Color', cmap(ii,:),'LineWidth',1)
        set(gca,'fontsize',14)
        hold on 
    end
    xLimits = get(gca,'XLim');    
    line(xLimits,[1,1],'Color','black','LineStyle','--')
    hold off
%    filename = ['./Figure/StepA'  '.png'];
%    print('-dpng', filename)   
end     
    
for sigma = 1:5
    figure(1001)
    subplot(5,1,sigma)
    ProbId = ResG{sigma,1}.ProbId;
    for ii = 1:5
        plot(ResG{sigma,1}.Alpha{ProbId(ii)}, 'Color', cmap(ii,:),'LineWidth',1)
        set(gca,'fontsize',14)
        hold on 
    end
    xLimits = get(gca,'XLim');    
    line(xLimits,[1,1],'Color','black','LineStyle','--')
    hold off
%    filename = ['./Figure/StepAG' '.png'];
%    print('-dpng', filename)
end     




    
       

        
            




