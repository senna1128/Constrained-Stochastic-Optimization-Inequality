%% This fucntion draws plot for simulation
clear all; close all
LenTest = [3,3,3,1];
%% Collect results for AdapNewton and AdapGD
load(['./Solution/AdapNewton', num2str(1),'.mat'])
AdapN = Result;
for IdSub = 2:4
    load(['./Solution/AdapNewton', num2str(IdSub),'.mat']) 
    AdapN = [AdapN; Result];
end
load(['./Solution/AdapGD', num2str(1),'.mat'])
AdapG = Result;
for IdSub = 2:4
    load(['./Solution/AdapGD', num2str(IdSub),'.mat']) 
    AdapG = [AdapG; Result];
end
Sigma = [1e-8,1e-4,1e-2,1e-1];


for Test = 1:4
    for id_Test = 1:LenTest(Test)
        for sigma = 1:4
            KKTVecN=[]; StdVecN=[]; CountFVecN=[]; CountGVecN=[]; CountHVecN=[];
            TimeVecN=[]; ProbIdN=[];
            FETriVecN=[]; FEOLDTriVecN=[]; NSRatioVecN=[]; FSRatioVecN=[];
            KKTVecG=[]; StdVecG=[]; CountFVecG=[]; CountGVecG=[]; CountHVecG=[];
            TimeVecG=[]; ProbIdG=[];
            FETriVecG=[]; FEOLDTriVecG=[]; NSRatioVecG=[]; FSRatioVecG=[];
            for IdProb = 1:39
                %% Newton
                ll = length(AdapN{IdProb}.KKTStep{Test}{id_Test,sigma});
                if ll > 0
                    ProbIdN = [ProbIdN; IdProb];
                    aaN=[];bbN=[];ccN=[];ddN=[];eeN=[];ffN=[];ggN=[];hhN=[];iiN=[];
                    for lll = 1:ll
                        if AdapN{IdProb}.KKTStep{Test}{id_Test,sigma}{lll}(end)<= 1.5   % consider only convergent case by chopping off some outliers
                            aaN = [aaN; min(AdapN{IdProb}.KKTStep{Test}{id_Test,sigma}{lll})];
                            [~, min_loc] = min(AdapN{IdProb}.KKTStep{Test}{id_Test,sigma}{lll});                            
                            bbN = [bbN; sum(max(min(cat(1,AdapN{IdProb}.CountFStep{Test}{id_Test,sigma}{lll}{1:min_loc}),Sigma(sigma)*1e7),1) )];  %% above sigma*e7 is regarded as true model   
                            ccN = [ccN; sum(max(min(cat(1,AdapN{IdProb}.CountGStep{Test}{id_Test,sigma}{lll}{1:min_loc}),Sigma(sigma)*1e7),1) )];   
                            ddN = [ddN; sum(max(min(cat(1,AdapN{IdProb}.CountHStep{Test}{id_Test,sigma}{lll}{1:min_loc}),Sigma(sigma)*1e7),1) )];
                            eeN = [eeN; AdapN{IdProb}.TimeStep{Test}{id_Test,sigma}{lll}(end)];
                            ffN = [ffN; AdapN{IdProb}.FETriStep{Test}{id_Test,sigma}{lll}(end)];
                            ggN = [ggN; AdapN{IdProb}.FEOLDTriStep{Test}{id_Test,sigma}{lll}(end)];
                            hhN = [hhN; AdapN{IdProb}.NSRatioStep{Test}{id_Test,sigma}{lll}(end)];
                            iiN = [iiN; AdapN{IdProb}.FSRatioStep{Test}{id_Test,sigma}{lll}(end)];
                        end
                    end
                    if ~isempty(aaN) && ~isnan(min(aaN)) && length(aaN)>=5   % average at least 5 independent runs
                        [AA,BB] = sort(aaN);
                        AA = AA(1:5); BB = BB(1:5); % pick first 5 runs
                        KKTVecN = [KKTVecN; mean(AA)];
                        StdVecN = [StdVecN; std(AA)];
                        CountFVecN = [CountFVecN; mean(bbN(BB))];
                        CountGVecN = [CountGVecN; mean(ccN(BB))];
                        CountHVecN = [CountHVecN; mean(ddN(BB))];
                        TimeVecN = [TimeVecN; mean(eeN(BB))];
                        FETriVecN = [FETriVecN; mean(ffN(BB))];
                        FEOLDTriVecN = [FEOLDTriVecN; mean(ggN(BB))];
                        NSRatioVecN = [NSRatioVecN; mean(hhN(BB))];
                        FSRatioVecN = [FSRatioVecN; mean(iiN(BB))];
                        [~,index] = min(aaN);
                        Res{Test}{sigma,id_Test}.Alpha{IdProb} = cat(1,AdapN{IdProb}.alphaStep{Test}{id_Test,sigma}{index}{:}); 
                        ll1 = length(AdapN{IdProb}.CountFStep{Test}{id_Test,sigma}{index});
                        ll2 = length(AdapN{IdProb}.CountGStep{Test}{id_Test,sigma}{index});
                        ll3 = length(AdapN{IdProb}.CountHStep{Test}{id_Test,sigma}{index});
                        ll = floor(min([ll1,ll2,ll3]));
                        Res{Test}{sigma,id_Test}.CountGFSeq{IdProb} = cat(1,AdapN{IdProb}.CountGStep{Test}{id_Test,sigma}{index}{1:ll})./cat(1,AdapN{IdProb}.CountFStep{Test}{id_Test,sigma}{index}{1:ll}); 
                        Res{Test}{sigma,id_Test}.CountHGSeq{IdProb} = cat(1,AdapN{IdProb}.CountHStep{Test}{id_Test,sigma}{index}{1:ll})./cat(1,AdapN{IdProb}.CountGStep{Test}{id_Test,sigma}{index}{1:ll}); 
                    end
                end
                %% GD
                ll = length(AdapG{IdProb}.KKTStep{Test}{id_Test,sigma});
                if ll > 0
                    ProbIdG = [ProbIdG; IdProb];
                    aaG=[];bbG=[];ccG=[];ddG=[];eeG=[];ffG=[];ggG=[];hhG=[];iiG=[];
                    for lll = 1:ll
                        if AdapG{IdProb}.KKTStep{Test}{id_Test,sigma}{lll}(end)<=1.5   % consider only convergent case by chopping off some outliers
                            aaG = [aaG; min(AdapG{IdProb}.KKTStep{Test}{id_Test,sigma}{lll})];
                            [~,min_loc] = min(AdapG{IdProb}.KKTStep{Test}{id_Test,sigma}{lll});
                            bbG = [bbG; sum(max(min(cat(1,AdapG{IdProb}.CountFStep{Test}{id_Test,sigma}{lll}{1:min_loc}),Sigma(sigma)*1e7),1) )];    
                            ccG = [ccG; sum(max(min(cat(1,AdapG{IdProb}.CountGStep{Test}{id_Test,sigma}{lll}{1:min_loc}),Sigma(sigma)*1e7),1) )];     
                            ddG = [ddG; sum(max(min(cat(1,AdapG{IdProb}.CountHStep{Test}{id_Test,sigma}{lll}{1:min_loc}),Sigma(sigma)*1e7),1) )];
                            eeG = [eeG; AdapG{IdProb}.TimeStep{Test}{id_Test,sigma}{lll}(end)];
                            ffG = [ffG; AdapG{IdProb}.FETriStep{Test}{id_Test,sigma}{lll}(end)];
                            ggG = [ggG; AdapG{IdProb}.FEOLDTriStep{Test}{id_Test,sigma}{lll}(end)];
                            hhG = [hhG; AdapG{IdProb}.NSRatioStep{Test}{id_Test,sigma}{lll}(end)];
                            iiG = [iiG; AdapG{IdProb}.FSRatioStep{Test}{id_Test,sigma}{lll}(end)];
                        end
                    end
                    if ~isempty(aaG) && ~isnan(min(aaG)) && length(aaG)>=5
                        [AA,BB] = sort(aaG);
                        AA = AA(1:5); BB = BB(1:5); % pick first 5 runs
                        KKTVecG = [KKTVecG; mean(AA)];
                        StdVecG = [StdVecG; std(AA)];
                        CountFVecG = [CountFVecG; mean(bbG(BB))];
                        CountGVecG = [CountGVecG; mean(ccG(BB))];
                        CountHVecG = [CountHVecG; mean(ddG(BB))];
                        TimeVecG = [TimeVecG; mean(eeG(BB))];
                        FETriVecG = [FETriVecG; mean(ffG(BB))];
                        FEOLDTriVecG = [FEOLDTriVecG; mean(ggG(BB))];
                        NSRatioVecG = [NSRatioVecG; mean(hhG(BB))];
                        FSRatioVecG = [FSRatioVecG; mean(iiG(BB))];
                        [~,index] = min(aaG);
                        ResG{Test}{sigma,id_Test}.Alpha{IdProb} = cat(1,AdapG{IdProb}.alphaStep{Test}{id_Test,sigma}{index}{:}); 
                        ll1 = length(AdapG{IdProb}.CountFStep{Test}{id_Test,sigma}{index});
                        ll2 = length(AdapG{IdProb}.CountGStep{Test}{id_Test,sigma}{index});
                        ll3 = length(AdapG{IdProb}.CountHStep{Test}{id_Test,sigma}{index});
                        ll = floor(min([ll1,ll2,ll3]));
                        ResG{Test}{sigma,id_Test}.CountGFSeq{IdProb} = cat(1,AdapG{IdProb}.CountGStep{Test}{id_Test,sigma}{index}{1:ll})./cat(1,AdapG{IdProb}.CountFStep{Test}{id_Test,sigma}{index}{1:ll}); 
                        ResG{Test}{sigma,id_Test}.CountHGSeq{IdProb} = cat(1,AdapG{IdProb}.CountHStep{Test}{id_Test,sigma}{index}{1:ll})./cat(1,AdapG{IdProb}.CountGStep{Test}{id_Test,sigma}{index}{1:ll}); 
                    end
                end
            end
            % Newton
            Res{Test}{sigma,id_Test}.KKT = KKTVecN;
            Res{Test}{sigma,id_Test}.CountF = CountFVecN;
            Res{Test}{sigma,id_Test}.CountG = CountGVecN;
            Res{Test}{sigma,id_Test}.CountH = CountHVecN;
            Res{Test}{sigma,id_Test}.Time = TimeVecN;
            Res{Test}{sigma,id_Test}.ProbId = ProbIdN;
            Res{Test}{sigma,id_Test}.Std = StdVecN;
            Res{Test}{sigma,id_Test}.FETri = FETriVecN;
            Res{Test}{sigma,id_Test}.FEOLDTri = FEOLDTriVecN;
            Res{Test}{sigma,id_Test}.NSRatio = NSRatioVecN;
            Res{Test}{sigma,id_Test}.FSRatio = FSRatioVecN;
            % GD
            ResG{Test}{sigma,id_Test}.KKT = KKTVecG;
            ResG{Test}{sigma,id_Test}.CountF = CountFVecG;
            ResG{Test}{sigma,id_Test}.CountG = CountGVecG;
            ResG{Test}{sigma,id_Test}.CountH = CountHVecG;
            ResG{Test}{sigma,id_Test}.Time = TimeVecG;
            ResG{Test}{sigma,id_Test}.ProbId = ProbIdG;
            ResG{Test}{sigma,id_Test}.Std = StdVecG;
            ResG{Test}{sigma,id_Test}.FETri = FETriVecG;
            ResG{Test}{sigma,id_Test}.FEOLDTri = FEOLDTriVecG;
            ResG{Test}{sigma,id_Test}.NSRatio = NSRatioVecG;
            ResG{Test}{sigma,id_Test}.FSRatio = FSRatioVecG;
        end
    end
end

save './Solution/AdapNewton.mat' Res
save './Solution/AdapGD.mat' ResG


