# This function estimates the function value of augmented lagrangian

function EstAugL(nlp,nx,necon,nicon,IdLow,IdUpp,cgrad,eta,epsilon,nu,sigma,CovM,Xi_f,Xi_grad2,f_k,nab_xL_k,G_k,c_k,X,MuLam,q_nuxlam,omega_epsnu_xlam,alpha_k,BarSQP,BV,diagg2lam)
   # estimate f
   bf_k = f_k+rand(Normal(0,(sigma/Xi_f)^(1/2)))
   bnab_xL_k = rand(MvNormal(nab_xL_k,CovM/Xi_grad2),1)
   J_1 = G_k[1:necon,:]*bnab_xL_k
   J_2 = G_k[IdUpp:end,:]*bnab_xL_k+diagg2lam
   baugL_k = bf_k+c_k'MuLam+1/(2*epsilon)*norm(c_k[1:necon])^2+1/(2*epsilon*q_nuxlam)*(norm(c_k[IdUpp:end])^2-norm(c_k[IdUpp:end]-omega_epsnu_xlam)^2)+eta/2*norm(J_1)^2+eta/2*norm(J_2)^2

   # estimate f_s
   # next iterate
   X_sk = X+alpha_k*BarSQP[1:nx]
   MuLam_sk = MuLam+alpha_k*BarSQP[nx+1:end]
   # objective value, gradient; constraint value, Jacobian
   f_sk, nabf_sk = objgrad(nlp,X_sk)
   un_c_sk, un_G_sk = consjac(nlp,X_sk)
   c_sk = [un_c_sk[nlp.meta.jfix];un_c_sk[nlp.meta.jupp]-BV[IdUpp:IdLow-1];-un_c_sk[nlp.meta.jlow]+BV[IdLow:end]]
   G_sk = [un_G_sk[nlp.meta.jfix,:];un_G_sk[nlp.meta.jupp,:];-un_G_sk[nlp.meta.jlow,:]]
   # intermediate terms
   G_sktMuLam_sk = G_sk'MuLam_sk
   diagg2lam_sk = ((c_sk.^2).*MuLam_sk)[IdUpp:end]
   a_snux = nu-sum(max.(c_sk[IdUpp:end],zeros(nicon)).^3)
   q_snuxlam = a_snux/(1+norm(MuLam_sk[IdUpp:end])^2)
   omega_epsnu_sxlam = max.(c_sk[IdUpp:end],-epsilon*q_snuxlam*MuLam_sk[IdUpp:end])
   # generate random quantities
   bf_sk = f_sk+rand(Normal(0,(sigma/Xi_f)^(1/2)))
   bnabf_sk = rand(MvNormal(nabf_sk,CovM/Xi_grad2),1)
   J_s1 = G_sk[1:necon,:]*(bnabf_sk+G_sktMuLam_sk)
   J_s2 = G_sk[IdUpp:end,:]*(bnabf_sk+G_sktMuLam_sk)+diagg2lam_sk
   baugL_sk = bf_sk+c_sk'MuLam_sk+1/(2*epsilon)*norm(c_sk[1:necon])^2+1/(2*epsilon*q_snuxlam)*(norm(c_sk[IdUpp:end])^2-norm(c_sk[IdUpp:end]-omega_epsnu_sxlam)^2)+eta/2*norm(J_s1)^2+eta/2*norm(J_s2)^2

   return baugL_k, baugL_sk
end
