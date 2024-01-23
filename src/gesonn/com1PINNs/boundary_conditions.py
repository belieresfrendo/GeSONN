def apply_BC(u, x, y, rho_min, rho_max, name=None):
    bc_mul = 0
    bc_add = 0
    if name=="dirichlet_homogene":
        rho_2 = x**2 + y**2
        bc_mul = rho_2 - rho_max**2
        if rho_min > 0:
            bc_mul = -bc_mul * (rho_min**2 - rho_2)
        bc_add = 0
    return u*bc_mul + bc_add