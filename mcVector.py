import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit, prange, set_num_threads
import time

#config.THREADING_LAYER = 'threadsafe'

# Created: 25 May 2023
# Last Update: 26 Dec 2023

#------------------------------------------
a = 5 # km
epsilon = 0.05
kappa = 0.5
zeta = 0.102
beta0 = 4 # km/s
alpha0 = beta0*np.sqrt(3) # km/s
n_shot = int(1e8)
nu = 0.8
fc = 3 # Hz
dx = 2 # km. Receiver volume is dx^3.
dt = 0.04 # sec
t_max = 60 # sec
dir_out = './env/py/a5e5k3z0.102/'
flag_wandering = True # wandering
#------------------------------------------

# set receiver location
n_rec = 6
rec_loc = np.zeros([n_rec,3])
for i in range(n_rec):
    r = (i+1)*25
    rec_loc[i,0] = r

# set parameters
nt = int(t_max/dt)
w0 = 2*np.pi*fc
k0 = w0/alpha0
l0 = w0/beta0
a_H = 1/(zeta*l0)
eta = a/a_H
vpvs = alpha0/beta0
gamma = math.gamma(kappa)
gamma05 = math.gamma(kappa+0.5)
gamma15 = math.gamma(kappa+1.5)
radiation_s = 1.0/(1.0+1.5*(vpvs**5)) # for radiation
if(kappa == 0.5):
    al0 = 2*(epsilon**2)*a*(1-1.0/(eta*eta))
    c_L = np.log(eta)
else:
    tmp_factor = (epsilon**2)*a*2*np.sqrt(np.pi)*gamma05/gamma
    al0 = tmp_factor*(1-(eta**(-2*kappa-1)))
    c_L = np.sqrt(np.pi)*gamma05*(1-(eta**(-2*kappa+1)))/(gamma*(2*kappa-1))
sigma_wzp = np.sqrt(al0*alpha0*dt) # for wandering
sigma_wzs = np.sqrt(al0*beta0*dt) # for wandering
sigma_alp = np.sqrt((2.0*(epsilon**2)/a)*c_L*alpha0*dt)
sigma_als = np.sqrt((2.0*(epsilon**2)/a)*c_L*beta0*dt)

# for the rejection method for the radiation
def cal_pdf_max_p():
    nth = 1000
    nphi = 1000
    dth = np.pi/nth
    dphi = 2.0*np.pi/nphi
    th1 = np.arange(nth)*dth
    th = np.tile(th1, nphi).reshape(nphi,nth).T
    phi1 = np.arange(nphi)*dphi
    phi = np.tile(phi1, nth).reshape(nth,nphi)
    pdf = (15.0/(16.0*np.pi))*(np.sin(th)**5)*(np.sin(2.0*phi)**2)
    return np.max(pdf)
def cal_pdf_max_s():
    nth = 1000
    nphi = 1000
    dth = np.pi/nth
    dphi = 2.0*np.pi/nphi
    th1 = np.arange(nth)*dth
    th = np.tile(th1, nphi).reshape(nphi,nth).T
    phi1 = np.arange(nphi)*dphi
    phi = np.tile(phi1, nth).reshape(nth,nphi)
    pdf = 5.0*(np.sin(2*th)**2)*(np.sin(2*phi)**2)/8.0 + 5.0*(np.sin(th)**2)*(np.cos(2*phi)**2)/2.0
    pdf *= np.sin(th)/(4*np.pi)
    return np.max(pdf)

# the max value for the rejection method is 1.05 times larger than the maximum
h_rad_p = 1.05*cal_pdf_max_p() 
h_rad_s = 1.05*cal_pdf_max_s()


@jit('Tuple((f8,f8))()', nopython=True)
def rad_angle_p():
    while(True):
        z = h_rad_p * np.random.rand()
        th = np.pi*np.random.rand()
        phi = 2*np.pi*np.random.rand()
        pdf = (15.0/(16.0*np.pi))*(np.sin(th)**5)*(np.sin(2.0*phi)**2)
        if(z <= pdf):
            break
    return th, phi
@jit('Tuple((f8,f8))()', nopython=True)
def rad_angle_s():
    while(True):
        z = h_rad_p * np.random.rand()
        th = np.pi*np.random.rand()
        phi = 2*np.pi*np.random.rand()
        pdf = 5.0*(np.sin(2*th)**2)*(np.sin(2*phi)**2)/8.0 + 5.0*(np.sin(th)**2)*(np.cos(2*phi)**2)/2.0
        pdf *= np.sin(th)/(4*np.pi)
        if(z <= pdf):
            break
    return th, phi
@jit('f8(f8,f8)', nopython=True)
def cal_upsilon(th, phi):
    b_th = 0.5*np.sin(2*th)*np.sin(2*phi)
    b_phi = np.sin(th)*np.cos(2*phi)
    upsilon = np.arctan2(b_phi, b_th)
    if(upsilon < 0):
        upsilon += 2*np.pi
    return upsilon

@jit('void(f8,f8,f8[:,:])', nopython=True)
def make_t_mat(th, phi, t_mat):
    cost = np.cos(th)
    sint = np.sin(th)
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    t_mat[0,0] = cost*cosp
    t_mat[0,1] = cost*sinp
    t_mat[0,2] = -sint
    t_mat[1,0] = -sinp
    t_mat[1,1] = cosp
    t_mat[1,2] = 0
    t_mat[2,0] = sint*cosp
    t_mat[2,1] = sint*sinp
    t_mat[2,2] = cost
@jit('void(f8, f8[:,:])', nopython=True)
def make_u_mat(upsilon, u_mat):
    cosu = np.cos(upsilon)
    sinu = np.sin(upsilon)
    u_mat[0,0] = cosu
    u_mat[0,1] = sinu
    u_mat[0,2] = 0
    u_mat[1,0] = -sinu
    u_mat[1,1] = cosu
    u_mat[1,2] = 0
    u_mat[2,0] = 0
    u_mat[2,1] = 0
    u_mat[2,2] = 1

@jit('Tuple((f8,f8))()', nopython=True)
def cal_angle_eikonal_p():
    random = np.random.rand()
    if(random == 1):
        random = np.random.rand()
    th = np.sqrt(-2.0*(sigma_alp*sigma_alp)*np.log(1.0-random))
    phi = 2.0*np.pi*np.random.rand()
    return th, phi
@jit('Tuple((f8,f8))()', nopython=True)
def cal_angle_eikonal_s():
    random = np.random.rand()
    if(random == 1):
        random = np.random.rand()
    th = np.sqrt(-2.0*(sigma_als*sigma_als)*np.log(1.0-random))
    phi = 2.0*np.pi*np.random.rand()
    return th, phi

def cal_psdf_h(m):
    psdf_h = (8*(np.pi**1.5)*gamma15*(epsilon*epsilon)*(a*a*a)/gamma) * ((eta*eta+a*a*m*m)**(-kappa-1.5))
    return psdf_h
@jit('f8(f8)', nopython=True)
def cal_psdf_h_scalar(m):
    # m is a scalar variable
    psdf_h = (8*(np.pi**1.5)*gamma15*(epsilon*epsilon)*(a*a*a)/gamma) * ((eta*eta+a*a*m*m)**(-kappa-1.5))
    return psdf_h
def cal_g_pp_r(th):
    gam2 = vpvs*vpvs
    sin2 = np.sin(th)*np.sin(th)
    x_pp_r = nu*(-1.0+np.cos(th)+2.0*sin2/gam2)-2.0 + 4.0*sin2/gam2
    x_pp_r /= gam2
    m = (2.0*l0/vpvs)*np.sin(0.5*th)
    psdf_h = cal_psdf_h(m)
    g_pp_r = ((l0*l0*l0*l0)/(4.0*np.pi))*(x_pp_r*x_pp_r)*psdf_h
    return g_pp_r
@jit('f8(f8)', nopython=True)
def cal_g_pp_r_scalar(th):
    gam2 = vpvs*vpvs
    sin2 = np.sin(th)*np.sin(th)
    x_pp_r = nu*(-1.0+np.cos(th)+2.0*sin2/gam2)-2.0 + 4.0*sin2/gam2
    x_pp_r /= gam2
    m = (2.0*l0/vpvs)*np.sin(0.5*th)
    psdf_h = cal_psdf_h_scalar(m)
    g_pp_r = ((l0*l0*l0*l0)/(4.0*np.pi))*(x_pp_r*x_pp_r)*psdf_h
    return g_pp_r
def cal_g_ps_t(th):
    cost = np.cos(th)
    x_ps_t = -np.sin(th)*(nu*(1.0-2.0*cost/vpvs)-4.0*cost/vpvs)
    m = (l0/vpvs)*np.sqrt(1.0+vpvs*vpvs-2.0*vpvs*np.cos(th))
    psdf_h = cal_psdf_h(m)
    g_ps_t = ((l0*l0*l0*l0)/(4.0*np.pi*vpvs))*(x_ps_t*x_ps_t)*psdf_h
    return g_ps_t
@jit('f8(f8)', nopython=True)
def cal_g_ps_t_scalar(th):
    cost = np.cos(th)
    x_ps_t = -np.sin(th)*(nu*(1.0-2.0*cost/vpvs)-4.0*cost/vpvs)
    m = (l0/vpvs)*np.sqrt(1.0+vpvs*vpvs-2.0*vpvs*np.cos(th))
    psdf_h = cal_psdf_h_scalar(m)
    g_ps_t = ((l0*l0*l0*l0)/(4.0*np.pi*vpvs))*(x_ps_t*x_ps_t)*psdf_h
    return g_ps_t
def cal_g_sp_r(th, phi):
    cost = np.cos(th)
    x_sp_r = nu*(1.0-2.0*cost/vpvs)-4.0*cost/vpvs
    x_sp_r *= np.sin(th)*np.cos(phi)/(vpvs*vpvs)
    m = (l0/vpvs)*np.sqrt(1.0+vpvs*vpvs-2.0*vpvs*np.cos(th))
    psdf_h = cal_psdf_h(m)
    g_sp_r = vpvs*((l0*l0*l0*l0)/(4.0*np.pi))*(x_sp_r*x_sp_r)*psdf_h
    return g_sp_r
@jit('f8(f8,f8)', nopython=True)
def cal_g_sp_r_scalar(th, phi):
    cost = np.cos(th)
    x_sp_r = nu*(1.0-2.0*cost/vpvs)-4.0*cost/vpvs
    x_sp_r *= np.sin(th)*np.cos(phi)/(vpvs*vpvs)
    m = (l0/vpvs)*np.sqrt(1.0+vpvs*vpvs-2.0*vpvs*np.cos(th))
    psdf_h = cal_psdf_h_scalar(m)
    g_sp_r = vpvs*((l0*l0*l0*l0)/(4.0*np.pi))*(x_sp_r*x_sp_r)*psdf_h
    return g_sp_r
def cal_g_ss_u(th, phi):
    cos2t = np.cos(2.0*th)
    cost = np.cos(th)
    x_ss_t = np.cos(phi)*(nu*(cost-cos2t)-2.0*cos2t)
    x_ss_p = np.sin(phi)*(nu*(cost-1.0)+2.0*cost)
    x_ss = np.sqrt(x_ss_t*x_ss_t + x_ss_p*x_ss_p)
    m = 2.0*l0*np.sin(0.5*th)
    psdf_h = cal_psdf_h(m)
    g_ss_u = ((l0*l0*l0*l0)/(4.0*np.pi))*(x_ss*x_ss)*psdf_h
    return g_ss_u
@jit('f8(f8,f8)', nopython=True)
def cal_g_ss_u_scalar(th, phi):
    cos2t = np.cos(2.0*th)
    cost = np.cos(th)
    x_ss_t = np.cos(phi)*(nu*(cost-cos2t)-2.0*cos2t)
    x_ss_p = np.sin(phi)*(nu*(cost-1.0)+2.0*cost)
    x_ss = np.sqrt(x_ss_t*x_ss_t + x_ss_p*x_ss_p)
    m = 2.0*l0*np.sin(0.5*th)
    psdf_h = cal_psdf_h_scalar(m)
    g_ss_u = ((l0*l0*l0*l0)/(4.0*np.pi))*(x_ss*x_ss)*psdf_h
    return g_ss_u
def cal_g_pp_r_0():
    dth = 0.001*np.pi/180.0
    nth = int(np.pi/dth)
    th = np.arange(nth)*dth
    g_pp_r = cal_g_pp_r(th)
    g_pp_r[-1] *= 0.5
    g0 = np.sum(g_pp_r*np.sin(th))*0.5*dth # 0.5 comes from 2pi/4pi
    return g0
def cal_g_ps_t_0():
    dth = 0.001*np.pi/180.0
    nth = int(np.pi/dth)
    th = np.arange(nth)*dth
    g_ps_t = cal_g_ps_t(th)
    g_ps_t[-1] *= 0.5
    g0 = np.sum(g_ps_t*np.sin(th))*0.5*dth # 0.5 comes from 2pi/4pi
    return g0
def cal_g_sp_r_0():
    nth = 2000
    nphi = 2000
    dth = np.pi/nth
    dphi = 2*np.pi/nphi
    th = np.arange(nth)*dth
    g_phi = np.zeros(nphi)
    for i in range(nphi):
        phi = i*dphi
        g_th = cal_g_sp_r(th, phi)
        g_th[0] *= 0.5
        g_th[-1] *= 0.5
        g_phi[i] = np.sum(g_th*np.sin(th))*dth
    g_phi[0] *= 0.5
    g_phi[-1] *= 0.5
    g0 = np.sum(g_phi)*dphi/(4*np.pi)
    return g0
def cal_g_ss_u_0():
    nth = 2000
    nphi = 2000
    dth = np.pi/nth
    dphi = 2*np.pi/nphi
    th = np.arange(nth)*dth
    g_phi = np.zeros(nphi)
    for i in range(nphi):
        phi = i*dphi
        g_th = cal_g_ss_u(th, phi)
        g_th[0] *= 0.5
        g_th[-1] *= 0.5
        g_phi[i] = np.sum(g_th*np.sin(th))*dth
    g_phi[0] *= 0.5
    g_phi[-1] *= 0.5
    g0 = np.sum(g_phi)*dphi/(4*np.pi)
    return g0

g_pp_0 = cal_g_pp_r_0()
g_ps_0 = cal_g_ps_t_0()
g_sp_0 = cal_g_sp_r_0()
g_ss_0 = cal_g_ss_u_0()
g_p_0 = g_pp_0 + g_ps_0
g_s_0 = g_ss_0 + g_sp_0
scat_p = g_p_0 * alpha0 * dt
scat_s = g_s_0 * beta0 * dt
scat_pp = g_pp_0 / g_p_0
scat_sp = g_sp_0 / g_s_0

def cal_pdf_max_born_pp():
    nth = 1000
    dth = np.pi/nth
    th = np.arange(nth)*dth
    pdf = cal_g_pp_r(th)/(4*np.pi*g_pp_0)
    return np.max(pdf)
def cal_pdf_max_born_ps():
    nth = 1000
    dth = np.pi/nth
    th = np.arange(nth)*dth
    pdf = cal_g_ps_t(th)/(4*np.pi*g_ps_0)
    return np.max(pdf)
def cal_pdf_max_born_sp():
    nth = 1000
    nphi = 1000
    dth = np.pi/nth
    dphi = 2.0*np.pi/nphi
    th1 = np.arange(nth)*dth
    th = np.tile(th1, nphi).reshape(nphi,nth).T
    phi1 = np.arange(nphi)*dphi
    phi = np.tile(phi1, nth).reshape(nth,nphi)
    pdf = cal_g_sp_r(th, phi)/(4*np.pi*g_sp_0)
    return np.max(pdf)
def cal_pdf_max_born_ss():
    nth = 1000
    nphi = 1000
    dth = np.pi/nth
    dphi = 2.0*np.pi/nphi
    th1 = np.arange(nth)*dth
    th = np.tile(th1, nphi).reshape(nphi,nth).T
    phi1 = np.arange(nphi)*dphi
    phi = np.tile(phi1, nth).reshape(nth,nphi)
    pdf = cal_g_ss_u(th, phi)/(4*np.pi*g_ss_0)
    return np.max(pdf)


h_born_pp = 1.05*cal_pdf_max_born_pp()
h_born_ps = 1.05*cal_pdf_max_born_ps()
h_born_sp = 1.05*cal_pdf_max_born_sp()
h_born_ss = 1.05*cal_pdf_max_born_ss()


@jit('Tuple((f8,f8))()', nopython=True)
def cal_angle_born_pp():
    phi = 2*np.pi*np.random.rand()
    factor = 1/(4*np.pi*g_pp_0)
    while(True):
        z = h_rad_p * np.random.rand()
        th = np.pi*np.random.rand()
        pdf = cal_g_pp_r_scalar(th)*factor
        if(z <= pdf):
            break
    return th, phi
@jit('Tuple((f8,f8))()', nopython=True)
def cal_angle_born_ps():
    phi = 2*np.pi*np.random.rand()
    factor = 1/(4*np.pi*g_ps_0)
    while(True):
        z = h_rad_p * np.random.rand()
        th = np.pi*np.random.rand()
        pdf = cal_g_ps_t_scalar(th)*factor
        if(z <= pdf):
            break
    return th, phi
@jit('Tuple((f8,f8))()', nopython=True)
def cal_angle_born_sp():
    factor = 1/(4*np.pi*g_sp_0)
    while(True):
        z = h_rad_p * np.random.rand()
        th = np.pi*np.random.rand()
        phi = 2*np.pi*np.random.rand()
        pdf = cal_g_sp_r_scalar(th, phi)*factor
        if(z <= pdf):
            break
    return th, phi
@jit('Tuple((f8,f8))()', nopython=True)
def cal_angle_born_ss():
    factor = 1/(4*np.pi*g_ss_0)
    while(True):
        z = h_rad_p * np.random.rand()
        th = np.pi*np.random.rand()
        phi = 2*np.pi*np.random.rand()
        pdf = cal_g_ss_u_scalar(th, phi)*factor
        if(z <= pdf):
            break
    return th, phi


  
@jit('i8(f8[:])', nopython=True)
def check_reach_receiver(xyz):
    flag = False
    for i in range(n_rec):
        for j in range(3):
            if((xyz[j]<rec_loc[i,j]-0.5*dx)|(xyz[j]>rec_loc[i,j]+0.5*dx)):
                flag = False
                break
            flag = True
        if(flag):
            return i
    return -1

parallel_options = {
    'comprehension': True,  # parallel comprehension
    'prange':        True,  # parallel for-loop
    'numpy':         True,  # parallel numpy calls
    'reduction':     True,  # parallel reduce calls
    'setitem':       True,  # parallel setitem
    'stencil':       True,  # parallel stencils
    'fusion':        False,  # enable fusion or not
}


@jit('f8[:,:](f8[:,:],f8[:,:])', nopython=True)
def dot_loop(mat1, mat2):
    mat3 = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            #mat3[i,j] = 0
            tmp = np.zeros(3)
            for k in range(3):
                tmp[k] = mat1[i,k]*mat2[k,j]
            mat3[i,j] = np.sum(tmp)
    return mat3

@jit('f8[:,:](f8[:,:],f8[:,:],f8[:,:])', nopython=True)
def dot3_loop(mat1, mat2, mat3):
    mat4 = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            tmp = np.zeros(9)
            for k in range(3):
                for l in range(3):
                    tmp[k*3+l] = mat1[i,k]*mat2[k,l]*mat3[l,j]
            mat4[i,j] = np.sum(tmp)
    return mat4
            



@jit('f8[:,:,:]()', nopython=True, parallel=parallel_options)
def run_mc():
    print('run')
    env = np.zeros((n_rec, 3, nt))
    for i in prange(n_shot):
        if(radiation_s > np.random.rand()):
            is_p = True
            th, phi = rad_angle_p()
            
        else:
            is_p = False
            th, phi = rad_angle_s()
            
            
        
        n_mat = np.zeros((3,3))
        for j in range(3):
            n_mat[j,j] = 1
        t_mat = np.zeros((3,3))
        u_mat = np.zeros((3,3))
        make_t_mat(th, phi, t_mat)

        if(is_p):
            n_mat = dot_loop(t_mat, n_mat)

        else:
            upsilon = cal_upsilon(th, phi)
            make_u_mat(upsilon, u_mat)
            n_mat = dot3_loop(u_mat, t_mat, n_mat)
            
        
        
        xyz = np.zeros(3) # particle position

        for j in prange(nt):
            # reach receiver?
            rec_index = check_reach_receiver(xyz)
            if(rec_index >= 0):
                if(is_p):
                    for k in prange(3):
                        env[rec_index,k,j] += n_mat[2,k]**2
                else:
                    for k in prange(3):
                        env[rec_index,k,j] += n_mat[0,k]**2
            # narrow angle scattering based on the Eikonal approx
            if(is_p):
                th, phi = cal_angle_eikonal_p()
            else:
                th, phi = cal_angle_eikonal_s()
            
            make_t_mat(th, phi, t_mat)
            if(is_p):
                n_mat = dot_loop(t_mat, n_mat)
            else:
                make_u_mat(-phi, u_mat)
                n_mat = dot3_loop(u_mat, t_mat, n_mat)


            # wide angle scattering based on the Born approx
            if(is_p):
                if(scat_p > np.random.rand()):
                    if(scat_pp > np.random.rand()):
                        th, phi = cal_angle_born_pp()
                        make_t_mat(th, phi, t_mat)
                        n_mat = dot_loop(t_mat, n_mat)
                    else:
                        th, phi = cal_angle_born_ps()
                        make_t_mat(th, phi, t_mat)
                        n_mat = dot_loop(t_mat, n_mat)
                        is_p = False
            else:
                if(scat_s > np.random.rand()):
                    if(scat_sp > np.random.rand()):
                        th, phi = cal_angle_born_sp()
                        make_t_mat(th, phi, t_mat)
                        n_mat = dot_loop(t_mat, n_mat)
                        is_p = True
                    else:
                        th, phi = cal_angle_born_ss()
                        make_t_mat(th, phi, t_mat)
                        cos2t = np.cos(2.0*th)
                        cost = np.cos(th)
                        x_ss_t = np.cos(phi)*(nu*(cost-cos2t)-2.0*cos2t)
                        x_ss_p = np.sin(phi)*(nu*(cost-1.0)+2.0*cost)
                        upsilon = np.arctan2(x_ss_p, x_ss_t)
                        if(upsilon < 0):
                            upsilon += 2*np.pi
                        make_u_mat(upsilon, u_mat)
                        n_mat = dot3_loop(u_mat, t_mat, n_mat)

            if(is_p):
                dist = dt*alpha0
                if(flag_wandering):
                    dist = np.random.normal(dt*alpha0, sigma_wzp)
            else:
                dist = dt*beta0
                if(flag_wandering):
                    dist = np.random.normal(dt*beta0, sigma_wzs)
            xyz += n_mat[2]*dist
    return env


        
start = time.time()
set_num_threads(10)
env = run_mc()
elapsed_time = time.time() - start
print ("run: {0}".format(elapsed_time) + "[sec]")

t = np.arange(nt)*dt
for i in range(n_rec):
    outname = dir_out + 'g_'+str(i)+'.txt'
    np.savetxt(outname, np.c_[t, env[i,0,:], env[i,1,:], env[i,2,:]])

plt.figure()
plt.plot(t, env[3,0,:])
plt.plot(t, env[3,1,:])
plt.plot(t, env[3,2,:])
plt.yscale('log')
plt.savefig('tmp.png')
print(np.max(env))

