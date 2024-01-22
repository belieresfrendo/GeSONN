def network_BC_mul(self, x, y):
    rho_2 = x**2 + y**2
    bc_mul = rho_2 - self.rho_max**2
    if self.rho_min > 0:
        bc_mul = -bc_mul * (self.rho_min**2 - rho_2)
    return bc_mul

def network_BC_add(self, x, y):
    return 0

def apply_BC(u, x, y, rho_min, rho_max, name=None):
    bc_mul = 1
    bc_add = 0
    if name=="dirichlet_homogene":
        rho_2 = x**2 + y**2
        bc_mul = rho_2 - rho_max**2
        if rho_min > 0:
            bc_mul = -bc_mul * (rho_min**2 - rho_2)
        bc_add = 0
    return u*bc_mul + bc_add