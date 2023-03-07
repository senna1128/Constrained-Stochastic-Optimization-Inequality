# This function estimates the function value of augmented lagrangian


function EstAugL(nx,ncon,X,MuLam,Id_Design,Id_Sig,Gau_Sigma,Exp_Sigma,eta,epsilon,nu,Xi_f,Xi_grad2,G_k,c_k,q_nuxlam,omega_epsnu_xlam,alpha_k,BarSQP,diagg2lam,G_ktMuLam_k,q_vector)
   Xi_f, Xi_grad2 = ceil(Int,Xi_f), ceil(Int,Xi_grad2)
   if Id_Design == 1
      # estimate f
      b1 = sign.(rand(Uniform(-1,1),Xi_f))
      A1 = rand(Normal(0,Gau_Sigma[Id_Sig]),Xi_f,nx)
      b2 = sign.(rand(Uniform(-1,1),Xi_grad2))
      A2 = rand(Normal(0,Gau_Sigma[Id_Sig]),Xi_grad2,nx)
   else
      b1 = sign.(rand(Uniform(-1,1),Xi_f))
      A1 = rand(Exponential(1/Exp_Sigma[Id_Sig]),Xi_f,nx).-1/Exp_Sigma[Id_Sig]
      b2 = sign.(rand(Uniform(-1,1),Xi_grad2))
      A2 = rand(Exponential(1/Exp_Sigma[Id_Sig]),Xi_grad2,nx).-1/Exp_Sigma[Id_Sig]
   end
   bf_k = sum(log.((1).+exp.(-b1.*A1*X)))/Xi_f
   bnabf_k = -sum(((1)./((1).+exp.(b2.*A2*X))).*b2.*A2,dims=1)/Xi_grad2
   bnab_xL_k = bnabf_k' + G_ktMuLam_k
   J_2 = G_k*bnab_xL_k+diagg2lam
   baugL_k = bf_k+c_k'MuLam+1/(2*epsilon*q_nuxlam)*(norm(c_k)^2-norm(c_k-omega_epsnu_xlam)^2)+eta/2*norm(J_2)^2
   # next iterate
   X_sk = X+alpha_k*BarSQP[1:nx]
   MuLam_sk = MuLam+alpha_k*BarSQP[nx+1:end]
   c_sk = G_k*X_sk+q_vector
   G_sktMuLam_sk = G_k'MuLam_sk
   diagg2lam_sk = (c_sk.^2).*MuLam_sk
   a_snux = nu-sum(max.(c_sk,zeros(ncon)).^3)
   q_snuxlam = a_snux/(1+norm(MuLam_sk)^2)
   omega_epsnu_sxlam = max.(c_sk,-epsilon*q_snuxlam*MuLam_sk)
   bf_sk = sum(log.((1).+exp.(-b1.*A1*X_sk)))/Xi_f
   bnabf_sk = -sum(((1)./((1).+exp.(b2.*A2*X_sk))).*b2.*A2,dims=1)/Xi_grad2
   J_s2 = G_k*(bnabf_sk'+G_sktMuLam_sk)+diagg2lam_sk
   baugL_sk = bf_sk+c_sk'MuLam_sk+1/(2*epsilon*q_snuxlam)*(norm(c_sk)^2-norm(c_sk-omega_epsnu_sxlam)^2)+eta/2*norm(J_s2)^2

   return baugL_k, baugL_sk
end
