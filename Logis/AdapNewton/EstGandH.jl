# This function estimate the gradient and Hessian of objective
### Obtain the estimate for bnabf_k, bnab2f_k


function EstGandH(nx,X,Id_Design,Id_Sig,Gau_Sigma,Exp_Sigma,G_ktMuLam_k,cgrad,DKKT_k,alpha_k,delta,kap_grad,chi_grad,p_grad,PrevSucc,MIN_VAR = 1e-6)
    # initial sample
    Xi_grad1 = 1
    # Different design
    if Id_Design == 1
        while true
            b = sign.(rand(Uniform(-1,1),Xi_grad1))
            A = rand(Normal(0,Gau_Sigma[Id_Sig]),Xi_grad1,nx)
            sigma = Gau_Sigma[Id_Sig]
            bnabf_k = -sum(((1)./((1).+exp.(b.*A*X))).*b.*A,dims=1)/Xi_grad1
            bnab_xL_k = bnabf_k' + G_ktMuLam_k
            barR_k = norm([bnab_xL_k;DKKT_k])
            if sigma/Xi_grad1 <= MIN_VAR
                Xi_H = ceil(Int, min(barR_k^2,1)*Xi_grad1)
                bb = sign.(rand(Uniform(-1,1),Xi_H))
                AA = rand(Normal(0,Gau_Sigma[Id_Sig]),Xi_H,nx)
                bnab_x2L_k = AA'*(((exp.(bb.*AA*X))./((1).+exp.(bb.*AA*X)).^2).*AA)
                return bnab_xL_k,bnab_x2L_k,Xi_grad1*1.0,Xi_H*1.0,barR_k
            else
                if PrevSucc == 1
                    Quant = min(kap_grad^2*alpha_k^2*barR_k^2,chi_grad^2*delta/alpha_k,1)
                else
                    Quant = min(kap_grad^2*alpha_k^2*barR_k^2,1)
                end
                if Xi_grad1 >= sigma*cgrad*log(nx/p_grad)/Quant
                    Xi_H = ceil(Int, min(barR_k^2,1)*Xi_grad1)
                    bb = sign.(rand(Uniform(-1,1),Xi_H))
                    AA = rand(Normal(0,Gau_Sigma[Id_Sig]),Xi_H,nx)
                    bnab_x2L_k = AA'*(((exp.(bb.*AA*X))./((1).+exp.(bb.*AA*X)).^2).*AA)
                    return bnab_xL_k,bnab_x2L_k,Xi_grad1*1.0,Xi_H*1.0,barR_k
                else
                    Xi_grad1 *= 100
                end
            end
        end
    else
        while true
            b = sign.(rand(Uniform(-1,1),Xi_grad1))
            A = rand(Exponential(1/Exp_Sigma[Id_Sig]),Xi_grad1,nx).-1/Exp_Sigma[Id_Sig]
            sigma = 1/Exp_Sigma[Id_Sig]^2
            bnabf_k = -sum(((1)./((1).+exp.(b.*A*X))).*b.*A,dims=1)/Xi_grad1
            bnab_xL_k = bnabf_k' + G_ktMuLam_k
            barR_k = norm([bnab_xL_k;DKKT_k])
            if sigma/Xi_grad1 <= MIN_VAR
                Xi_H = ceil(Int, min(barR_k^2,1)*Xi_grad1)
                bb = sign.(rand(Uniform(-1,1),Xi_H))
                AA = rand(Exponential(1/Exp_Sigma[Id_Sig]),Xi_H,nx).-1/Exp_Sigma[Id_Sig]
                bnab_x2L_k = AA'*(((exp.(bb.*AA*X))./((1).+exp.(bb.*AA*X)).^2).*AA)
                return bnab_xL_k,bnab_x2L_k,Xi_grad1*1.0,Xi_H*1.0,barR_k
            else
                if PrevSucc == 1
                    Quant = min(kap_grad^2*alpha_k^2*barR_k^2,chi_grad^2*delta/alpha_k,1)
                else
                    Quant = min(kap_grad^2*alpha_k^2*barR_k^2,1)
                end
                if Xi_grad1 >= sigma*cgrad*log(nx/p_grad)/Quant
                    Xi_H = ceil(Int, min(barR_k^2,1)*Xi_grad1)
                    bb = sign.(rand(Uniform(-1,1),Xi_H))
                    AA = rand(Exponential(1/Exp_Sigma[Id_Sig]),Xi_H,nx).-1/Exp_Sigma[Id_Sig]
                    bnab_x2L_k = AA'*(((exp.(bb.*AA*X))./((1).+exp.(bb.*AA*X)).^2).*AA)
                    return bnab_xL_k,bnab_x2L_k,Xi_grad1*1.0,Xi_H*1.0,barR_k
                else
                    Xi_grad1 *= 100
                end
            end
        end
    end
end
