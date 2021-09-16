%% This fucntion draws plot for simulation
%{
clear all; close all
d = dir('./Solution/*.mat');
A = dir(fullfile('./Figure/*'));
if ~isempty(A)
    for k = 1:length(A)
        delete(strcat('./Figure/', A(k).name))
    end
    fprintf('remove result file. Done!\n')
else
    fprintf('No result file.\n')
end
%}
%% load file
for i = 1:length(d)
    load(['./Solution/',d(i).name])
end
%% Extract AdapNewton result
for sigma = 1:5
    KKTVec = [];
    TimeVec = [];
    CountVec = [];
    for Idprob = 1:23
        ll = length(AdapN{Idprob}.KKTStep{sigma});
        if ll > 0
            a = [];
            b = [];
            c = [];
            for lll = 1:ll 
                a = [a; AdapN{Idprob}.KKTStep{sigma}{lll}(end)];
                b = [b; AdapN{Idprob}.CountStep{sigma}{lll}(end)];
                c = [c; AdapN{Idprob}.TimeStep{sigma}{lll}(end)];
            end
            KKTVec = [KKTVec; min(a)];
            CountVec = [CountVec; min(b)];
            TimeVec = [TimeVec; min(c)];
        end 
    end 
    AdapNKKT{sigma} = KKTVec;
    AdapNCount{sigma} = CountVec;
    AdapNTime{sigma} = TimeVec;
end


%% Extract AdapGD result
for sigma = 1:5
    KKTVec = [];
    TimeVec = [];
    CountVec = [];
    for Idprob = 1:23
        ll = length(AdapG{Idprob}.KKTStep{sigma});
        if ll > 0
            a = [];
            b = [];
            c = [];
            for lll = 1:ll 
                a = [a; AdapG{Idprob}.KKTStep{sigma}{lll}(end)];
                b = [b; AdapG{Idprob}.CountStep{sigma}{lll}(end)];
                c = [c; AdapG{Idprob}.TimeStep{sigma}{lll}(end)];
            end
            KKTVec = [KKTVec; min(a)];
            CountVec = [CountVec; min(b)];
            TimeVec = [TimeVec; min(c)];
        end 
    end 
    AdapGKKT{sigma} = KKTVec;
    AdapGCount{sigma} = CountVec;
    AdapGTime{sigma} = TimeVec;
end



%% Extract nonadaptive result 
for ConStep = 1:4
    for sigma = 1:5 
        KKTVec = [];
        TimeVec = [];
        CountVec = [];
        for Idprob = 1:39
            ll = length(NonAdapR{Idprob}.KKTCStep{ConStep,sigma});
            if ll > 0 
                a = [];
                b = [];
                c = [];
                for lll = 1:ll 
                    a = [a; NonAdapR{Idprob}.KKTCStep{ConStep,sigma}{lll}(end)];
                    b = [b; NonAdapR{Idprob}.TimeCStep{ConStep,sigma}{lll}(end)];
                    c = [c; length(NonAdapR{Idprob}.KKTCStep{ConStep,sigma}{lll})];
                end
                if ~isnan(min(a))
                    KKTVec = [KKTVec; min(a)];
                    TimeVec = [TimeVec; min(b)];
                    CountVec = [CountVec; min(c)];
                end
            end 
        end
        NonAdapCKKT{ConStep,sigma} = KKTVec;
        NonAdapCTime{ConStep,sigma} = TimeVec;
        NonAdapCCount{ConStep,sigma} = CountVec;
    end 
end

for DecayStep = 1:2
    for sigma = 1:5 
        KKTVec = [];
        TimeVec = [];
        CountVec = [];
        for Idprob = 1:39
            ll = length(NonAdapR{Idprob}.KKTDStep{DecayStep,sigma});
            if ll > 0 
                a = [];
                b = [];
                c = [];
                for lll = 1:ll 
                    a = [a; NonAdapR{Idprob}.KKTDStep{DecayStep,sigma}{lll}(end)];
                    b = [b; NonAdapR{Idprob}.TimeDStep{DecayStep,sigma}{lll}(end)];
                    c = [c; length(NonAdapR{Idprob}.KKTDStep{DecayStep,sigma}{lll})];
                end
                if ~isnan(min(a))
                    KKTVec = [KKTVec; min(a)];
                    TimeVec = [TimeVec; min(b)];
                    CountVec = [CountVec; min(c)];
                end
            end 
        end
        NonAdapDKKT{DecayStep,sigma} = KKTVec;
        NonAdapDTime{DecayStep,sigma} = TimeVec;
        NonAdapDCount{DecayStep,sigma} = CountVec;
    end 
end



%% Plot
% Go over constant stepsize 
for ConStep = 1:4
    % Plot KKT residual
    data = cell(5, 3);
    for sigma = 1:size(data,1)
        Ac{sigma} = AdapNKKT{sigma};
        Bc{sigma} = AdapGKKT{sigma};
        if length(NonAdapCKKT{ConStep, sigma})>0 
            Cc{sigma} = NonAdapCKKT{ConStep, sigma};
        else 
            Cc{sigma} = [NaN];
        end
    end
    data = vertcat(Ac,Bc,Cc);
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    col=[102,255,255, 200;
        51,153,255, 200;
        0, 0, 255, 200];
    col=col/255;
    multiple_boxplot(data',xlab,{'AdapNewton', 'AdapGD', 'NonAdap'},col',1)
    filename = ['./Figure/KKTCon' num2str(ConStep) '.png'];
    print('-dpng', filename)
    
end

% Go over decay steps

for DecayStep = 1:2
    % Plot KKT residual
    data = cell(5, 3);
    for sigma = 1:size(data,1)
        Ac{sigma} = AdapNKKT{sigma};
        Bc{sigma} = AdapGKKT{sigma};
        if length(NonAdapDKKT{DecayStep, sigma})>0 
            Cc{sigma} = NonAdapDKKT{DecayStep, sigma};
        else 
            Cc{sigma} = [NaN];
        end
    end
    data = vertcat(Ac,Bc,Cc);
    xlab={'1e-8','1e-4','1e-2','1e-1','1'};
    col=[102,255,255, 200;
        51,153,255, 200;
        0, 0, 255, 200];
    col=col/255;
    multiple_boxplot(data',xlab,{'AdapNewton', 'AdapGD', 'NonAdap'},col',1)
    filename = ['./Figure/KKTDecay' num2str(DecayStep) '.png'];
    print('-dpng', filename)
    
end

% Plot consuming time

data = cell(5,2);
for sigma = 1:size(data,1)
    Ac{sigma} = AdapNTime{sigma};
    Bc{sigma} = AdapGTime{sigma};
end
data = vertcat(Ac,Bc);
multiple_boxplot(data',xlab,{'AdapNewton', 'AdapGD'},col(2:3,:)',2)
filename = ['./Figure/Time.png'];
print('-dpng', filename)

    
  
       

        
            




