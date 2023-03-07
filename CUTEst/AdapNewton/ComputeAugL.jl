## This function computes the augmented Lagrangian and its related
## quantities, given a set of epsilon, nu, eta

function ComputeAugL(necon,epsilon,eta,c_k,MuLam,a_nux,q_nuxlam,diagg2lam,J_21,bnab_xL_k,Q_1,J_1,Q_2,G_ktl,G_k,M_k)
    # Inequality constraints
    cc_k, lam_k = c_k[necon+1:end], MuLam[necon+1:end]
    # Compute omega_epsnu_xlam
    omega_epsnu_xlam = max.(cc_k, -epsilon*q_nuxlam*lam_k)
    # Construct active set
    act_set = findall(cc_k.>= -epsilon*q_nuxlam*lam_k)
    # Compute J_2c
    Pcdiagg2lam = diagg2lam
    Pcdiagg2lam[act_set] = zeros(length(act_set))
    J_2c = J_21 + Pcdiagg2lam
    # Compute augmented Lagrangian gradient
    bnab_xaugL_k1 = bnab_xL_k+eta*Q_1*J_1+eta*Q_2*J_2c+1/epsilon*G_k'[:,1:necon]*c_k[1:necon]+1/(epsilon*q_nuxlam)*G_k'[:,necon+1:end]*omega_epsnu_xlam
    bnab_xaugL_k = bnab_xaugL_k1+3*norm(omega_epsnu_xlam)^2/(2*epsilon*a_nux*q_nuxlam)*G_ktl+eta*Q_2[:,act_set]*diagg2lam[act_set]
    bnab_mlaugL_k1 = [c_k[1:necon];omega_epsnu_xlam]+eta*M_k*[J_1;J_2c]
    bnab_mlaugL_k = bnab_mlaugL_k1+[zeros(necon);norm(omega_epsnu_xlam)^2/(epsilon*a_nux)*lam_k] + eta*M_k[:,act_set.+necon]*diagg2lam[act_set]
    bnab_augL_k=[bnab_xaugL_k; bnab_mlaugL_k]
    bnab_augL_k1=[bnab_xaugL_k1; bnab_mlaugL_k1]
    return bnab_augL_k,bnab_augL_k1,J_2c,omega_epsnu_xlam,act_set
end
