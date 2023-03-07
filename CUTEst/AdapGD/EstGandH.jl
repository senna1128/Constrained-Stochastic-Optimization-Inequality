# This function estimate the gradient and Hessian of objective

function EstGandH(nx,nab_xL_k,nab_x2L_k,cgrad,DKKT_k,sigma,CovM,alpha_k,delta,kap_grad,chi_grad,p_grad,PrevSucc,MIN_VAR = 1e-4)
    # initial sample
    Xi_grad1 = 1.0
    # go over while loop
    while true
        bnab_xL_k = rand(MvNormal(nab_xL_k,CovM/Xi_grad1),1)
        barR_k = norm([bnab_xL_k;DKKT_k])
        if sigma/Xi_grad1 <= MIN_VAR
            Xi_H = ceil(min(barR_k^2,1)*Xi_grad1)
            Delta = rand(Normal(0,(sigma/Xi_H)^(1/2)),nx,nx)
            bnab_x2L_k = Hermitian(nab_x2L_k,:L)+(Delta+Delta')/2
            return bnab_xL_k, bnab_x2L_k, Xi_grad1, Xi_H, barR_k
        else
            if PrevSucc == 1
                Quant = min(kap_grad^2*alpha_k^2*barR_k^2,chi_grad^2*delta/alpha_k,1)
            else
                Quant = min(kap_grad^2*alpha_k^2*barR_k^2,1)
            end
            if Xi_grad1 >= cgrad*log(nx/p_grad)/Quant
                Xi_H = ceil(min(barR_k^2,1)*Xi_grad1)
                Delta = rand(Normal(0,(sigma/Xi_H)^(1/2)),nx,nx)
                bnab_x2L_k = Hermitian(nab_x2L_k,:L)+(Delta+Delta')/2
                return bnab_xL_k, bnab_x2L_k, Xi_grad1, Xi_H, barR_k
            else
                Xi_grad1 *= 5
            end
        end
    end
end
