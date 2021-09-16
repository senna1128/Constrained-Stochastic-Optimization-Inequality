# This function estimate the gradient and Hessian of objective

function EstGandH(nx,Xi,nabf_k,nab2f_k,G_ktMuLam_k,DKKT_k,sigma,CovM,alpha_k,kap_grad,p_grad,C_grad,MAX_SAMPLE = 100000)
    while true
        bnabf_k = mean(rand(MvNormal(nabf_k, CovM), convert(Int64, floor(Xi))), dims=2)
        Delta = rand(Normal(0,(sigma/Xi)^(1/2)), nx, nx)
        bnab2f_k = Hermitian(nab2f_k, :L) + (Delta + Delta')/2

        if Xi >= MAX_SAMPLE
            return bnabf_k, bnab2f_k, Xi
        else
            barR_k = norm([bnabf_k+G_ktMuLam_k;DKKT_k])
            Quant = min(kap_grad^2*alpha_k^2*barR_k^2,1)
            if Xi >= C_grad*log(nx/p_grad)/Quant
                return bnabf_k, bnab2f_k, Xi
            else
                Xi *= 1.5
            end
        end
    end
end
