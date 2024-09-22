import numpy as np
from numba import njit, prange
import time
import numba as nb

# Realization number
nreal = 1

lbox = 1000.
ngrid = 256

flux_filename = '...'

dtype = np.float64


# Parameters for power spectrum computation
convert_to_delta = False   # If ouput of webon, set to False, as webon produces already the overdensty field

write_pk = False
outpk = 'pk_rspace_recon_%d.txt' %nreal

nbins_pk = ngrid//2


# Bias for Zel'dovich approximation
b1 = 1.
b2 = 1.
b3 = 1.
b4 = 1.

# Growth factor
ff = 0.5

# LOS
los = 2

# Reconstruction mode (0: isotropic; 1: anistropic)
reconmode = 0

# Enable RSD? 0 if real space, 1 if redshift space. Other values for redshift space introduce velocity bias
rsdfact = 0.

# **********************************************
# **********************************************
# **********************************************
def fft(inarr,ngrid):
    finarr= np.fft.fftn(inarr)

    return finarr

def ifft(finarr,ngrid):
    inarr= np.fft.ifftn(finarr)

    return inarr

# **********************************************
def measure_spectrum(signal):

    nbin = int(round(nbins_pk))
    
    fsignal = np.fft.fftn(signal) #np.fft.fftn(signal)

    kmax = np.pi * ngrid / lbox #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))
    dk = kmax/nbin  # Bin width

    nmode = np.zeros((nbin))
    kmode = np.zeros((nbin))
    power = np.zeros((nbin))

    kmode, power, nmode = get_power(fsignal, nbin, kmax, dk, kmode, power, nmode)
    
    return kmode[1:], power[1:]


# **********************************************                                                                                         
@njit(parallel=False, cache=True)
def get_power(fsignal, Nbin, kmax, dk, kmode, power, nmode):
    
    for i in prange(ngrid):
        for j in prange(ngrid):
            for k in prange(ngrid):
                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,i,j,k))
                if ktot <= kmax:
                    nbin = int(ktot/dk-0.5)
                    akl = fsignal.real[i,j,k]
                    bkl = fsignal.imag[i,j,k]
                    kmode[nbin]+=ktot
                    power[nbin]+=(akl*akl+bkl*bkl)
                    nmode[nbin]+=1

    for m in prange(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            power[m]/=nmode[m]

    power = power / (ngrid/2)**3

    return kmode, power, nmode

@njit(cache=True)
def k_squared_nohermite(lbox,ngrid,ii,jj,kk):

      kfac = 2.0*np.pi/lbox

      if ii <= ngrid/2:
        kx = kfac*ii
      else:
        kx = -kfac*(ngrid-ii)

      if jj <= ngrid/2:
        ky = kfac*jj
      else:
        ky = -kfac*(ngrid-jj)

      if kk <= ngrid/2:
          kz = kfac*kk
      else:
          kz = -kfac*(ngrid-kk)                                                                                                           

      k2 = kx**2+ky**2+kz**2

      return k2

# **********************************************
@njit(cache=True)
def k_squared(lbox,ngrid,ii,jj,kk):
    
      kfac = 2.0*np.pi/lbox

      if ii <= ngrid/2:
        kx = kfac*ii
      else:
        kx = -kfac*(ngrid-ii)
      
      if jj <= ngrid/2:
        ky = kfac*jj
      else:
        ky = -kfac*(ngrid-jj)
      
      #if kk <= nc/2:
      kz = kfac*kk
      #else:
      #  kz = -kfac*np.float64(nc-k)
      
      k2 = kx**2+ky**2+kz**2

      return k2


# **********************************************
@njit(parallel=True, cache=True)
def GradFinDiff(lbox,ngrid,arr,dim):

    fac = ngrid/(2*lbox)

    outarr = arr.copy()

    for xx in prange(ngrid):
        for yy in range(ngrid):
            for zz in range(ngrid):

                xdummy = np.array([xx,xx,xx,xx])
                ydummy = np.array([yy,yy,yy,yy])
                zdummy = np.array([zz,zz,zz,zz])
                xxr = xdummy[0]
                xxrr = xdummy[1]
                xxl = xdummy[2]
                xxll = xdummy[3]
                yyr = ydummy[0]
                yyrr = ydummy[1]
                yyl = ydummy[2]
                yyll = ydummy[3]
                zzr = zdummy[0]
                zzrr = zdummy[1]
                zzl = zdummy[2]
                zzll = zdummy[3]

                # Periodic BCs
                if dim == 1:
                    xxl = xx - 1
                    xxll = xx - 2
                    xxr = xx + 1
                    xxrr = xx + 2
                    
                    if xxl<0:
                        xxl += ngrid
                    if xxl>=ngrid:
                        xxl -= ngrid
                    
                    if xxll<0:
                        xxll += ngrid
                    if xxll>=ngrid:
                        xxll -= ngrid
                    
                    if xxr<0:
                        xxr += ngrid
                    if xxr>=ngrid:
                        xxr -= ngrid

                    if xxrr<0:
                        xxrr += ngrid
                    if xxrr>=ngrid:
                        xxrr -= ngrid


                elif dim == 2:
                    
                    yyl = yy - 1
                    yyll = yy - 2
                    yyr = yy + 1
                    yyrr = yy + 2
                    
                    if yyl<0:
                        yyl += ngrid
                    if yyl>=ngrid:
                        yyl -= ngrid
                    
                    if yyll<0:
                        yyll += ngrid
                    if yyll>=ngrid:
                        yyll -= ngrid
                    
                    if yyr<0:
                        yyr += ngrid
                    if yyr>=ngrid:
                        yyr -= ngrid

                    if yyrr<0:
                        yyrr += ngrid
                    if yyrr>=ngrid:
                        yyrr -= ngrid


                elif dim == 3:
                    
                    zzl = zz - 1
                    zzll = zz - 2
                    zzr = zz + 1
                    zzrr = zz + 2
                    
                    if zzl<0:
                        zzl += ngrid
                    if zzl>=ngrid:
                        zzl -= ngrid
                    
                    if zzll<0:
                        zzll += ngrid
                    if zzll>=ngrid:
                        zzll -= ngrid
                    
                    if zzr<0:
                        zzr += ngrid
                    if zzr>=ngrid:
                        zzr -= ngrid

                    if zzrr<0:
                        zzrr += ngrid
                    if zzrr>=ngrid:
                        zzrr -= ngrid

                outarr[xx,yy,zz] = -fac*((4.0/3.0)*(arr[xxl,yyl,zzl]-arr[xxr,yyr,zzr])-(1.0/6.0)*(arr[xxll,yyll,zzll]-arr[xxrr,yyrr,zzrr]))

    return outarr

# **********************************************
@njit(parallel=True, cache=True)
def Tweb(arr, ngrid, lbox):

    # Get gradients exploiting simmetry of the tensor, i.e. gradxy=gradyx

    # X DIRECTION
    # 1st deriv
    grad = GradFinDiff(lbox,ngrid,arr,1)
    #2nd derivs
    gradxx = GradFinDiff(lbox,ngrid,grad,1)
    gradxy = GradFinDiff(lbox,ngrid,grad,2)
    gradxz = GradFinDiff(lbox,ngrid,grad,3)

    # Y DIRECTION
    # 1st deriv
    grad = GradFinDiff(lbox,ngrid,arr,2)
    #2nd derivs
    gradyy = GradFinDiff(lbox,ngrid,grad,2)
    gradyz = GradFinDiff(lbox,ngrid,grad,3)

    # Z DIRECTION
    # 1st deriv
    grad = GradFinDiff(lbox,ngrid,arr,3)
    #2nd derivs
    gradzz = GradFinDiff(lbox,ngrid,grad,3)

    #del arr, grad

    lambda1 = np.zeros_like((gradxx))
    lambda2 = np.zeros_like((gradxx))
    lambda3 = np.zeros_like((gradxx))
    tweb = np.zeros_like((gradxx))

    # Compute eigenvalues    
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):
                mat = np.array([[gradxx[ii,jj,kk],gradxy[ii,jj,kk],gradxz[ii,jj,kk]],[gradxy[ii,jj,kk],gradyy[ii,jj,kk],gradyz[ii,jj,kk]],[gradxz[ii,jj,kk],gradyz[ii,jj,kk],gradzz[ii,jj,kk]]])
                eigs = np.linalg.eigvals(mat)
                eigs = np.flip(np.sort(eigs))
                lambda1[ii,jj,kk] = eigs[0]
                lambda2[ii,jj,kk] = eigs[1]
                lambda3[ii,jj,kk] = eigs[2]
                if eigs[0]>=0 and eigs[1]>=0 and eigs[2]>=0:
                    tweb[ii,jj,kk] = 1
                elif eigs[0]>=0 and eigs[1]>=0 and eigs[2]<0:
                    tweb[ii,jj,kk] = 2
                elif eigs[0]>=0 and eigs[1]<0 and eigs[2]<0:
                    tweb[ii,jj,kk] = 3
                elif eigs[0]<0 and eigs[1]<0 and eigs[2]<0:
                    tweb[ii,jj,kk] = 4

    # Now compute invariants
    #del gradxx, gradxy, gradxz, gradyy,gradyz,gradzz
    
    #I1 = lambda1 + lambda2 + lambda3
    #I2 = lambda1 * lambda2 + lambda1 * lambda3 + lambda2 * lambda3
    #I3 = lambda2 * lambda2 * lambda3

    #del lambda1, lambda2, lambda3

    return tweb

# **********************************************
@njit(parallel=True, cache=True, fastmath=True)
def RankOrder(ngrid, arrin, arrtg):

    arrin = arrin.flatten()

    # Prepare the fields
    arrtg_cp = arrtg.copy()
    arrtg_cp = np.sort(arrtg_cp)

    arrin_new = arrin.copy()

    arrin_ind = np.argsort(arrin_new)

    for ii in prange(ngrid**3):

        arrin_new[arrin_ind[ii]] = arrtg_cp[ii]
    
    arrin_new = np.reshape(arrin_new, (ngrid,ngrid,ngrid))
    
    return arrin_new

# **********************************************
def ZeldovichApproximation(ngrid, lbox, delta, tweb, ff, b1, b2, b3, b4, reconmode):

    '''
    @ Compute the displacement field using the Zel'dovich approximation
    '''

    deltaf = fft(delta, ngrid**3)
    
    kfac = 2.0*np.pi/lbox
    
    psixf, psiyf, psizf = Zeldovich_loop_numba(ngrid, lbox, deltaf, kfac, tweb, ff, b1, b2, b3, b4, reconmode)

    psix = ifft(psixf,ngrid**3).real
    psiy = ifft(psiyf,ngrid**3).real
    psiz = ifft(psizf,ngrid**3).real
    
    return psix, psiy, psiz

@njit(parallel=True, cache=True)
def Zeldovich_loop_numba(ngrid, lbox, deltaf, kfac, tweb, ff, b1, b2, b3, b4, reconmode):

    # reconmode = 0 ---> isotropic (Padmanabhan et al. 2012)
    # reconmode = 1 ---> isotropic (Eisenstein et al. 2007)

    psixf = np.zeros((ngrid,ngrid,ngrid), dtype=nb.complex128)
    psiyf = np.zeros((ngrid,ngrid,ngrid), dtype=nb.complex128)
    psizf = np.zeros((ngrid,ngrid,ngrid), dtype=nb.complex128)

    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                # Find Tweb-dependent bias parameters
                if tweb[ii,jj,kk] == 1.:
                    bb = b1
                elif tweb[ii,jj,kk] == 2.:
                    bb = b2
                elif tweb[ii,jj,kk] == 3.:
                    bb = b3
                elif tweb[ii,jj,kk] == 4.:
                    bb = b4

                # Compute mu
                kpar, kper = get_kpar_kper(lbox,ngrid,ii,jj,kk)

                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,ii,jj,kk))

                # find the value of mu
                if ii==0 and jj==0 and kk==0:  
                    mu = 0.0
                else:    
                    mu = kpar/ktot

                # Compute beta and the normalization factor for the kernel 
                beta = ff / bb

                if reconmode == 0: # isotropic
                    fact = bb * (1. + beta*mu**2)
                else: # anisotropic
                    fact = bb

                if ii <= ngrid/2:
                    kx = kfac*np.float64(ii)
                else:
                    kx = -kfac*np.float64(ngrid-ii)
                if jj <= ngrid/2:
                    ky = kfac*np.float64(jj)
                else:
                    ky = -kfac*np.float64(ngrid-jj)
                if kk <= ngrid/2:
                    kz = kfac*np.float64(kk)
                else:
                    kz = -kfac*np.float64(ngrid-kk)

                k2 = kx**2 + ky**2 + kz**2

                kernelx = 0.
                kernely = 0.
                kernelz = 0.


                # kernel: -nabla/nabla2 = i*kvec/k2 comes from: 
                # d(exp(i*kvec*r))/dr=i*kvec  , d(exp(i*kvec*r))2/dr2=(i*kvec)*(i*kvec)=-k2 

                epsk = 1e-14
                if k2>epsk:
                    kernelx = kx/(k2*fact)
                    kernely = ky/(k2*fact)
                    kernelz = kz/(k2*fact)
                
                psixf.real[ii,jj,kk] = -kernelx*deltaf.imag[ii,jj,kk]
                psixf.imag[ii,jj,kk] =  kernelx*deltaf.real[ii,jj,kk]

                psiyf.real[ii,jj,kk] = -kernely*deltaf.imag[ii,jj,kk]
                psiyf.imag[ii,jj,kk] =  kernely*deltaf.real[ii,jj,kk]

                psizf.real[ii,jj,kk] = -kernelz*deltaf.imag[ii,jj,kk]
                psizf.imag[ii,jj,kk] =  kernelz*deltaf.real[ii,jj,kk]

    return psixf, psiyf, psizf

# **********************************************
@njit(cache=True)
def get_kpar_kper(lbox,ngrid,ii,jj,kk):

    kfac = 2.0*np.pi/lbox

    if ii <= ngrid/2:
        kx = kfac*ii
    else:
        kx = -kfac*(ngrid-ii)

    if jj <= ngrid/2:
        ky = kfac*jj
    else:
        ky = -kfac*(ngrid-jj)

    if kk <= ngrid/2:
          kz = kfac*kk
    else:
          kz = -kfac*(ngrid-kk)

    # compute the value of k_par and k_perp
    if los==0:   
        k_par, k_per = kx, np.sqrt(ky*ky + kz*kz)
    elif los==1: 
        k_par, k_per = ky, np.sqrt(kx*kx + kz*kz)
    else:         
        k_par, k_per = kz, np.sqrt(kx*kx + ky*ky)
                                                                                                               
    return k_par, k_per

# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def GetCic(posx, posy, posz, weight, lbox, ngrid):

    weight = weight.flatten()

    lcell = lbox/ngrid

    delta = np.zeros((ngrid,ngrid,ngrid))

    for ii in range(len(posx)):
        xx = posx[ii]
        yy = posy[ii]
        zz = posz[ii]
        indxc = int(xx/lcell)
        indyc = int(yy/lcell)
        indzc = int(zz/lcell)

        wxc = xx/lcell - indxc
        wyc = yy/lcell - indyc
        wzc = zz/lcell - indzc

        if wxc <=0.5:
            indxl = indxc - 1
            if indxl<0:
                indxl += ngrid
            wxc += 0.5
            wxl = 1 - wxc
        elif wxc >0.5:
            indxl = indxc + 1
            if indxl>=ngrid:
                indxl -= ngrid
            wxl = 1 - wxc

        if wyc <=0.5:
            indyl = indyc - 1
            if indyl<0:
                indyl += ngrid
            wyc += 0.5
            wyl = 1 - wyc
        elif wyc >0.5:
            indyl = indyc + 1
            if indyl>=ngrid:
                indyl -= ngrid
            wyl = 1 - wyc

        if wzc <=0.5:
            indzl = indzc - 1
            if indzl<0:
                indzl += ngrid
            wzc += 0.5
            wzl = 1 - wzc
        elif wzc >0.5:
            indzl = indzc + 1
            if indzl>=0:
                indzl -= ngrid
            wzl = 1 - wzc

        ww = weight[ii]

        delta[indxc,indyc,indzc] += ww * wxc * wyc * wzc
        delta[indxl,indyc,indzc] += ww * wxl * wyc * wzc
        delta[indxc,indyl,indzc] += ww * wxc * wyl * wzc
        delta[indxc,indyc,indzl] += ww * wxc * wyc * wzl
        delta[indxl,indyl,indzc] += ww * wxl * wyl * wzc
        delta[indxc,indyl,indzl] += ww * wxc * wyl * wzl
        delta[indxl,indyc,indzl] += ww * wxl * wyc * wzl
        delta[indxl,indyl,indzl] += ww * wxl * wyl * wzl

    delta = delta/np.mean(delta) - 1.

    return delta

# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def DisplaceParticles(ngrid, lbox, psix, psiy, psiz, tweb, ff, b1, b2, b3, b4, reconmode, rsdfact):

    # CONVENTION: the Zel'dovich displacement have been computed in the forward direction, hence we will subtract them 

    # reconmode = 0 ---> isotropic (Padmanabhan et al. 2012)
    # reconmode = 1 ---> isotropic (Eisenstein et al. 2007)

    lcell = lbox / ngrid

    posx = np.zeros((ngrid**3))
    posy = np.zeros((ngrid**3))
    posz = np.zeros((ngrid**3))

    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                posxtmp = (ii + 0.5) * lcell
                posytmp = (jj + 0.5) * lcell
                posztmp = (kk + 0.5) * lcell

                ind3d = kk + ngrid*(jj + ngrid*ii)

                # Find Tweb-dependent bias parameters
                if tweb[ii,jj,kk] == 1.:
                    bb = b1
                elif tweb[ii,jj,kk] == 2.:
                    bb = b2
                elif tweb[ii,jj,kk] == 3.:
                    bb = b3
                elif tweb[ii,jj,kk] == 4.:
                    bb = b4

                beta = ff / bb

                if reconmode == 0: # isotropic
                    fact = ff
                else: # anisotropic
                    fact = (ff-beta)/(1.+beta)

                # Displace particles
                posx[ind3d] = posxtmp - psix[ii,jj,kk] 
                posy[ind3d] = posytmp - psiy[ii,jj,kk] 
                posz[ind3d] = posztmp - (1. + rsdfact * fact) * psiz[ii,jj,kk]  

    return posx, posy, posz

# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def DisplaceRandoms(ngrid, lbox, psix, psiy, psiz, tweb, ff, b1, b2, b3, b4, reconmode, rsdfact):

    # CONVENTION: the Zel'dovich displacement have been computed in the forward direction, hence we will subtract them  
    
    # reconmode = 0 ---> isotropic (Padmanabhan et al. 2012)
    # reconmode = 1 ---> isotropic (Eisenstein et al. 2007)

    lcell = lbox / ngrid

    posx = np.zeros((ngrid**3))
    posy = np.zeros((ngrid**3))
    posz = np.zeros((ngrid**3))

    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                posxtmp = (ii + 0.5) * lcell
                posytmp = (jj + 0.5) * lcell
                posztmp = (kk + 0.5) * lcell

                ind3d = kk + ngrid*(jj + ngrid*ii)

                # Find Tweb-dependent bias parameters
                if tweb[ii,jj,kk] == 1.:
                    bb = b1
                elif tweb[ii,jj,kk] == 2.:
                    bb = b2
                elif tweb[ii,jj,kk] == 3.:
                    bb = b3
                elif tweb[ii,jj,kk] == 4.:
                    bb = b4

                beta = ff / bb

                if reconmode == 0: # isotropic
                    fact = 1.
                else: # anisotropic
                    fact = (ff-beta)/(1.+beta)

                # Displace particles
                posx[ind3d] = posxtmp - psix[ii,jj,kk] 
                posy[ind3d] = posytmp - psiy[ii,jj,kk] 
                posz[ind3d] = posztmp - (1. + rsdfact * fact) * psiz[ii,jj,kk]  

    return posx, posy, posz

# **********************************************
def write_pk(kk, pk, outpk):

    ff = open(outpk, 'w')

    for ii in range(len(kk)):
        ff.write(str(kk[ii]) + '      ' + str(pk[ii]) + '\n')

    ff.close()

# **********************************************
# **********************************************
# **********************************************
# MAIN

# Start timing 
ti = time.time()

# READ INPUT
print('Reading input files ...')
#flux = np.fromfile(flux_filename, dtype=dtype)
#flux = np.reshape(flux, (ngrid,ngrid,ngrid))

# Pass from flux to optical depth
#tau = -np.log10(flux)

tau = np.random.normal(0,1,size=(ngrid,ngrid,ngrid))

print('... done!')
print('')

# Now rank-order the optical depth to a Gaussian field
# First, create the Gaussina distribution (NB: it's a Gaussian distribution, NOT a Gaussian random field with a P(k))
print('Rank-ordering the tau field to a Gaussian distribution ...')
gausstg = np.random.normal(0., 1., size=ngrid**3)
deltagauss = RankOrder(ngrid, tau, gausstg)
print('... done!')
print('')

# Here, normalize by the biases in the CW environments
# First perform the CWC of the gaussianised density field
print('Computing T-web ...')
tweb = Tweb(deltagauss, ngrid, lbox)
print('... done!')
print('')

# Compute the displacement field
print("Computing Zel'dovich displacements ...")
psix, psiy, psiz = ZeldovichApproximation(ngrid, lbox, deltagauss, tweb, ff, b1, b2, b3, b4, reconmode)
print('... done!')
print('')

# Displace true "particles" (i.e. pixels)
print('Applying the displacemets backward in time and computing the reconstructed field ...')
xd, yd, zd = DisplaceParticles(ngrid, lbox, psix, psiy, psiz, tweb, ff, b1, b2, b3, b4, reconmode, rsdfact)
deltad = GetCic(xd, yd, zd, tau, lbox, ngrid)

# Displace random particles
xs, ys, zs = DisplaceRandoms(ngrid, lbox, psix, psiy, psiz, tweb, ff, b1, b2, b3, b4, reconmode, rsdfact)
deltas = GetCic(xs, ys, zs, tau, lbox, ngrid)

# Compute reconstructed density
deltarec = deltad - deltas

print('... done!')
print('')

# Compute power spectrum
print('Measuring P(k) ...')
kk, pk = measure_spectrum(deltarec)

# Write power spectrum to file
if write_pk==True:
    print('... and writing P(k) to file ...')
    write_pk(kk,pk,outpk)

print('... done!')
print('')

tf = time.time()
dt = tf - ti

uu = 'seconds'

if dt>60.:
    dt /= 60.
    uu = 'minutes'
    if dt > 60.:
        dt /= 60.
        uu = 'hours'

print('The end. Elapsed ' + str(dt) + ' ' + uu + '.')