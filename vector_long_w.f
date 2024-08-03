C***********************************************************************
      program vector_long_w
c
c     Created 19 Aug 2019 by K. Emoto
c     Last update: 22 Dec 2021
c
c     export OMP_NUM_THREADS=10
c     source /opt/intel/oneapi/setvars.sh
c     mpif90 vector_long_w.f mt19937.f -shared-intel -mcmodel=large -O3 -o vector_long_w.out
c     nohup mpirun -n 10 ./vector_long_w.out > vector_long_w.log &
c
C***********************************************************************

      implicit none
      include 'mpif.h'

      double precision grnd
      external grnd

      integer ii, jj, kk, ll, mm, itmp, itmp1, itmp2
      integer seed, nofst, q, nt, stindex, scatangleindex, nofst1
      integer ndist
      integer(8) nofshot, nofshot1
      double precision tmax, a, eps, kap, zeta, nu
      double precision gamma0, beta0, alpha0, gam, gam05, gam15
      double precision k0, pi, f0, fq, tq
      double precision ag0, dt0, dxpos, w0, el0, aH, epsH
      double precision dtmp, dtmp1, dtmp2, dtmp3, dtmp4, dtmp5
      double precision thl, phil, thll, phill, d, r2, v
      double precision sradfac
      double precision dsincos, dsinsin, dcosine, dsine
      double precision hPDFRadP, sigmaLP, CLp, gP0, gS0, sigmaLS, CLs
      double precision CL
      double precision gPP0, gPS0, gSS0, gSP0, sigma
      double precision scatfacp, scatfacs, hRp, hRs
      double precision scatfacpp, scatfacss, tmpscatfac, scatfacsp
      double precision hHPP, hHPS, hHSP, hHSS, vXSSt, vXSSp
      double precision rr, tt, pp, stx, sty, stz, stth, stphi
      double precision twp, tws, a0, tfluc, alp0, als0, al0, eta
      double precision sigwzp, sigwzs, zfluc
      integer np, myid, nproc, ibuf, istat, ierr
      integer calstart, calend, calrate, icnt
      real cpustart, cpuend
      character outdir*70
      logical isP, flagtmp, flagwan
      parameter(seed=6500) !500 1000 1500 2000 2500 3000 3500 4000
      parameter(nofshot=5e8) !5e8
      parameter(nofst=24) ! nofst1 * ndist
      parameter(nofst1=4)
      parameter(ndist=6) 
      parameter (gamma0=1.7320508)
      parameter (beta0=4.0)
      parameter (alpha0=6.928)
      parameter (nu=0.8)
      parameter(a=5.0)
      parameter(eps=0.05)
      parameter(kap=0.5)
      parameter(zeta=0.102)
c      parameter (stdist=50.0)
      parameter (np=10)
      parameter(f0=3.0) !Hz
      parameter(dxpos=1.0) !km 0.5
      parameter(dt0=0.04, tmax=40) !s 0.04
      parameter(nt=int(tmax/dt0))
      parameter(ibuf=nt*nofst*3)
      double precision propangle(2), xyz(3)
      double precision env(nofst,3,nt)
      double precision stpos(nofst,3), stposmin(nofst,3)
      double precision stposmax(nofst,3), stpos1(nofst1,3)
      double precision rbuf(nt,3,nofst,np-1), sbuf(nt,3,nofst)
      double precision envpa(nt)
      double precision nmat(3,3)
      double precision tmat(3,3), ymat(3,3)
      integer ntq, nfq, itq, ifq
      parameter(ntq=100, nfq=200)
      double precision dtq, dfq, th
      double precision distr(nofshot, 2), hist(ntq, nfq)
      double precision symth(16), symphi(16)
      double precision stdist(ndist), stdist1
      character filename*80
      outdir = "./env/env_a5e5k5f3z0.102_sd_13/"

      flagtmp = .false.
      flagwan = .false. !wandering

      call system_clock(calstart)
      call cpu_time(cpustart)

      call MPI_INIT     (ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, nproc, ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)

      if ( nproc .ne. np ) then
         CALL  MPI_FINALIZE ( ierr )
         WRITE(6,*) '## NP ERROR ##'
          write(*,*) 'nproc, np ',nproc,np
         STOP
      endif

      nofshot1 = nofshot / np
      if(myid .eq. 0) then
        nofshot1 = nofshot1 + mod(nofshot, np)
      end if
      write(6,*) 'myid, nofshot1', myid, nofshot1

      pi = acos(-1.0_16)
      w0 = 2.0*pi*f0
      k0 = w0/alpha0
      el0 = w0/beta0
      epsH = eps*((zeta*a*el0)**(-kap))
      aH = 1.0/(zeta*el0)
      eta = a/aH
      gam05 = exp(gammln(kap+0.5))
      gam15 = exp(gammln(kap+1.5))
      gam = exp(gammln(kap))
      sradfac = 1.0/(1.0+1.5*(gamma0**5))
      a0 = 2*sqrt(pi)*eps*eps*a*gam05/gam

      if(kap .eq. 0.5) then
        !alp0 = 2*eps*eps*a*(1-1.0/((zeta*a*k0)**2))
        !als0 = 2*eps*eps*a*(1-1.0/((zeta*a*el0)**2))
        al0 = 2*eps*eps*a*(1-1.0/(eta*eta))
      else
        dtmp = eps*eps*a*2*sqrt(pi)*gam05/gam
        al0 = dtmp*(1-(eta**(-2*kap-1)))
        !alp0 = dtmp*(1-((zeta*a*k0)**(-2*kap-1)))
        !als0 = dtmp*(1-((zeta*a*el0)**(-2*kap-1)))
      end if

      !twp = sqrt(alp0*dt0/alpha0)
      !tws = sqrt(als0*dt0/beta0)
      sigwzp = sqrt(al0*alpha0*dt0)
      sigwzs = sqrt(al0*beta0*dt0)
      

      dtq = pi/dble(ntq)
      dfq = 2.0*pi/dble(nfq)
      do ii=1, ntq
        do jj=1, nfq
          hist(ii,jj) = 0.0d0
        end do
      end do

      call sgrnd(seed+myid)

      !make receiver location
      stdist(1) = 25.0
      stdist(2) = 50.0
      stdist(3) = 75.0
      stdist(4) = 100.0
      stdist(5) = 125.0
      stdist(6) = 150.0
      !stdist(7) = 175.0
      !stdist(8) = 200.0

      itmp = 1
      
      do ii=1, ndist
        call makeSymStPosRTP(stdist(ii), 90.0d0, 0.0d0, stpos1, pi)
        do jj=1, nofst1
          stpos(itmp, 1:3) = stpos1(jj, 1:3)
          itmp = itmp + 1
        end do

      end do
      do ii=1, nofst
        do jj=1, 3
          stposmin(ii,jj) = stpos(ii,jj)-0.5*dxpos
          stposmax(ii,jj) = stpos(ii,jj)+0.5*dxpos
        end do
      end do
      ! if(myid .eq. 0) then
      !   do ii=1, nofst
      !     write(6,*) ii, stpos(ii,1), stpos(ii,2), stpos(ii,3)
      !   end do
      ! end if


      hRp = 1.1*maxPDFRp(pi)
      hRs = 1.1*maxPDFRs(pi)
      CL = fCL(gam, gam05, kap, eta, pi)
      !CLp = fCLp(gam, gam05, kap, a, zeta, k0, pi)
      sigmaLP = sqrt((2.0*eps*eps/a)*CL*alpha0*dt0)
      !CLs = fCLs(gam, gam05, kap, a, zeta, el0, pi)
      sigmaLS = sqrt((2.0*eps*eps/a)*CL*beta0*dt0)
      gPP0 = fgPPr0(el0,pi,gamma0,gam,gam15,a,eps,kap,eta,nu)
      gPS0 = fgPSt0(el0,pi,gamma0,gam,gam15,a,eps,kap,eta,nu)
      gSP0 = fgSPr0(el0,pi,gamma0,gam,gam15,a,eps,kap,eta,nu)
      gSS0 = fgSSs10(el0,pi,gamma0,gam,gam15,a,eps,kap,eta,nu)
      gP0 = gPP0 + gPS0
      gS0 = gSS0 + gSP0
      scatfacp = gP0*alpha0*dt0
      scatfacs = gS0*beta0*dt0
      scatfacpp = gPP0/gP0
      scatfacss = gSS0/gS0
      scatfacsp = gSP0/gS0
      tmpscatfac = gPP0*alpha0*dt0
      dtmp = log(10.0)
      if(myid .eq. 0) then
        write(6,*) 'log', dtmp
        write(6,*) 'f0,w0,k0,el0,zeta', f0,w0,k0,el0,zeta
        write(6,*) 'epsH,aH,dt0', epsH, aH, dt0
        write(6,*) 'CL', CL
        write(6,*) 'sigmaLP', sigmaLP
        write(6,*) 'sigmaLS', sigmaLS
        write(6,*) 'scatfacss', scatfacss
        write(6,*) 'scatfacpp', scatfacpp
        write(6,*) 'tmpscatfac', tmpscatfac
        write(6,*) 'gP0', gP0
        write(6,*) 'gPP0', gPP0
        write(6,*) 'gPS0', gPS0
        write(6,*) 'gS0', gS0
        write(6,*) 'gSP0', gSP0
        write(6,*) 'gSS0', gSS0
        write(6,*) 'gP0 alpha0 dt', scatfacp
        write(6,*) 'gS0 beta0 dt', scatfacs
        write(6,*) 'sradfac', sradfac
      end if

      hHPP = 1.1*maxPDFHPPr(pi,gamma0,gam,gam15,a,eps,kap,eta,
     :      gPP0,nu)
      hHPS = 1.1*maxPDFHPSt(pi,gamma0,gam,gam15,a,eps,kap,eta,
     :      gPS0,nu)
      hHSP = 1.1*maxPDFHSPr(pi,gamma0,gam,gam15,a,eps,kap,eta,
     :      gSP0,nu)
      hHSS = 1.1*maxPDFHSSs1(pi,gamma0,gam,gam15,a,eps,kap,eta,
     :      gSS0,nu)
      if(myid .eq. 0) then
        write(6,*) 'hHPP', hHPP
        write(6,*) 'hHPS', hHPS
        write(6,*) 'hHSP', hHSP
        write(6,*) 'hHSS', hHSS
        write(6,*) 'hRp', hRp
        write(6,*) 'hRs', hRs
      end if

      do kk=1, nt
        do jj=1, 3
          do ii=1, nofst
            env(ii,jj,kk) = 0.0
          end do
        end do
      end do


      do ii=1, nofshot1
        if(sradfac .gt. grnd()) then
          isP = .true.
          call frRp(pi, tq, fq, hRp)
        else
          isP = .false.
          call frRs(pi, tq, fq, hRs)
        end if
c        tq = acos(1.0-2.0*grnd())
c        fq = 2.0*pi*grnd()
c        isP = .true.
        do jj=1, 3
          do kk=1, 3
            nmat(kk,jj) = 0.0
          end do
        end do
        nmat(1,1) = 1
        nmat(2,2) = 1
        nmat(3,3) = 1
        tmat = makeTmat(tq, fq)
        if(isP) then
          call calProdMat(tmat, nmat)
        else
          v = atanRad(tq, fq, pi)
          ymat = makeYmat(v)
          call calProdMatYT(ymat, tmat, nmat)
        end if

        xyz(1) = 0.0
        xyz(2) = 0.0
        xyz(3) = 0.0
        do jj=1, nt
          stindex = checkReachStation(xyz,stposmin,stposmax,
     :                nofst)
          if(stindex .gt. 0) then
            if(isP) then
              do kk=1, 3
                env(stindex,kk,jj) = env(stindex,kk,jj) +
     :            nmat(3,kk)**2
c                 env(stindex,kk,jj) = env(stindex,kk,jj)+1
              end do
            else
              do kk=1, 3
                env(stindex,kk,jj) = env(stindex,kk,jj) +
     :            nmat(1,kk)**2
c                 env(stindex,kk,jj) = env(stindex,kk,jj)+1
              end do
            end if
          end if

          if(isP) then
            call rLP(pi, tq, fq, sigmaLP)
          else
            call rLS(pi, tq, fq, sigmaLS)
          end if
          tmat = makeTmat(tq, fq)
          if(isP) then
           call calProdMat(tmat, nmat)
          else
            ymat = makeYmat(-fq)
            call calProdMatYT(ymat, tmat, nmat)
          end if

          if(isP) then
            if(scatfacp .gt. grnd()) then
              if(scatfacpp .gt. grnd()) then
                call rHPP(el0,pi,gamma0,gam,gam15,a,eps,kap,eta,
     :                gPP0,tq,fq,nu,hHPP)
c              if(tmpscatfac .gt. grnd()) then
c                tq = acos(1.0-2.0*grnd())
c                fq = 2.0*pi*grnd()
                tmat = makeTmat(tq, fq)
                call calProdMat(tmat, nmat)
              else
                call rHPS(el0,pi,gamma0,gam,gam15,a,eps,kap,eta,
     :                gPS0,tq,fq,nu,hHPS)
                tmat = makeTmat(tq, fq)
                call calProdMat(tmat, nmat)
                isP = .false.
              end if
            end if
          else
            if(scatfacs .gt. grnd()) then
              if(scatfacsp .gt. grnd()) then
                call rHSP(el0,pi,gamma0,gam,gam15,a,eps,kap,eta,
     :               gSP0,tq,fq,nu,hHSP)
                tmat = makeTmat(tq, fq)
                call calProdMat(tmat, nmat)
                isP = .true.
              else
                call rHSS(el0,pi,gamma0,gam,gam15,a,eps,kap,eta,
     :               gSS0,tq,fq,nu,hHSS)
                tmat = makeTmat(tq, fq)
                vXSSt = XSSt(tq, fq, nu)
                vXSSp = XSSp(tq, fq, nu)
                sigma = atan3(vXSSp, vXSSt, pi)
                ymat = makeYmat(sigma)
                call calProdMatYT(ymat, tmat, nmat)
              end if
            end if
          end if

          zfluc = 0
          if(isP) then
            if(flagwan) then
              zfluc = randnorm(pi, dt0*alpha0, sigwzp)
            else
              zfluc = alpha0*dt0
            end if
          else
            if(flagwan) then
              zfluc = randnorm(pi, dt0*beta0, sigwzs)
            else
              zfluc = beta0*dt0
            end if
          end if
          do kk=1, 3
            xyz(kk) = xyz(kk) + nmat(3,kk)*zfluc
          end do


        end do !time

      end do !shot

      do kk=1, nofst
        do jj=1, 3
          do ii=1, nt
            sbuf(ii,jj,kk) = env(kk,jj,ii)
          end do
        end do
      end do

      call mpi_barrier(mpi_comm_world,ierr)

      if(myid .eq. 0) then
        do ii=1, np-1
          call mpi_recv(rbuf(1,1,1,ii),ibuf,mpi_double_precision
     :       ,ii,ii,mpi_comm_world,istat,ierr)
        end do
      else
        call mpi_send(sbuf(1,1,1),ibuf,mpi_double_precision
     :       ,0,myid,mpi_comm_world,ierr)
      end if

      if(myid .eq. 0) then
        do ll=1, np-1
          do kk=1, nofst
            do jj=1, 3
              do ii=1, nt
                env(kk,jj,ii) = env(kk,jj,ii) + rbuf(ii,jj,kk,ll)
              end do
            end do
          end do
        end do
        call makeEnv(env, nofshot, nofst, dxpos, nt)
        call fileOutGrs(outdir, env, nofst, dt0, nt)

      end if

      call mpi_finalize(ierr)

      call system_clock(calend, calrate)
      call cpu_time(cpuend)
      write(*,*) 'calculation time:', 1*(calend-calstart)/calrate
      write(*,*) 'cpu time:', cpuend-cpustart

      contains

      subroutine makeSymStPos(xr, yr, zr, stpos1, pi)
        implicit none
        double precision, intent(in) :: xr, yr, zr, pi
        double precision, intent(inout) :: stpos1(16,3)
        double precision th, phi, thr, phir, r
        r = sqrt(xr**2+yr**2)
        phir = acos(xr/r)
        if (r .le. 0.001) then
          phir = 0
        end if
        r = sqrt(xr**2+yr**2+zr**2)
        thr = acos(zr/r)
        stpos1(1, 1:3) = stxyz(r, thr, phir)
        stpos1(2, 1:3) = stxyz(r, thr, 0.5*pi-phir)
        stpos1(3, 1:3) = stxyz(r, thr, 0.5*pi+phir)
        stpos1(4, 1:3) = stxyz(r, thr, pi-phir)
        stpos1(5, 1:3) = stxyz(r, thr, pi+phir)
        stpos1(6, 1:3) = stxyz(r, thr, 1.5*pi-phir)
        stpos1(7, 1:3) = stxyz(r, thr, 1.5*pi+phir)
        stpos1(8, 1:3) = stxyz(r, thr, 2.0*pi-phir)
        stpos1(9, 1:3) = stxyz(r, pi-thr, phir)
        stpos1(10, 1:3) = stxyz(r, pi-thr, 0.5*pi-phir)
        stpos1(11, 1:3) = stxyz(r, pi-thr, 0.5*pi+phir)
        stpos1(12, 1:3) = stxyz(r, pi-thr, pi-phir)
        stpos1(13, 1:3) = stxyz(r, pi-thr, pi+phir)
        stpos1(14, 1:3) = stxyz(r, pi-thr, 1.5*pi-phir)
        stpos1(15, 1:3) = stxyz(r, pi-thr, 1.5*pi+phir)
        stpos1(16, 1:3) = stxyz(r, pi-thr, 2.0*pi-phir)
      end subroutine

      subroutine makeSymStPosRTP(str, stth_d, stph_d, stpos1, pi)
        implicit none
        double precision, intent(in) :: str, stth_d, stph_d, pi
        double precision, intent(inout) :: stpos1(4,3)
        double precision th, phi, thr, phir, r
        phir = stph_d * pi / 180.0 !acos(xr/r)
        r = str !sqrt(xr**2+yr**2+zr**2)
        thr = stth_d * pi / 180.0 !acos(zr/r)
        stpos1(1, 1:3) = stxyz(r, thr, phir)
        stpos1(2, 1:3) = stxyz(r, thr, 0.5*pi)
        stpos1(3, 1:3) = stxyz(r, thr, pi)
        stpos1(4, 1:3) = stxyz(r, thr, 1.5*pi)
      end subroutine
        

      function stxyz(r, th, phi)
        implicit none
        double precision stxyz(3)
        double precision r, th, phi, sint
        sint = sin(th)
        stxyz(1) = r*sint*cos(phi)
        stxyz(2) = r*sint*sin(phi)
        stxyz(3) = r*cos(th)
      end function

      function PvK(m, gam, gam15, a, eps, kap, pi)
        implicit none
        double precision gam, gam15, a, eps, kap, pi, m
        double precision PvK
        PvK = 8.0*(pi**1.5)*gam15*(eps**2)*(a**3)/gam
        PvK = PvK * ((1.0+a*a*m*m)**(-kap-1.5))
      end function

      function makeTmat(tq, fq)
        implicit none
        double precision, intent(in) :: tq, fq
        double precision makeTmat(3,3)
        double precision cost, sint, cosp, sinp
        cost = cos(tq)
        sint = sin(tq)
        cosp = cos(fq)
        sinp = sin(fq)
        makeTmat(1,1) = cost*cosp
        makeTmat(1,2) = cost*sinp
        makeTmat(1,3) = -sint
        makeTmat(2,1) = -sinp
        makeTmat(2,2) = cosp
        makeTmat(2,3) = 0
        makeTmat(3,1) = sint*cosp
        makeTmat(3,2) = sint*sinp
        makeTmat(3,3) = cost
      end function

      function makeYmat(x)
        implicit none
        double precision, intent(in) :: x
        double precision makeYmat(3,3)
        double precision cosx, sinx
        cosx = cos(x)
        sinx = sin(x)
        makeYmat(1,1) = cosx
        makeYmat(1,2) = sinx
        makeYmat(1,3) = 0
        makeYmat(2,1) = -sinx
        makeYmat(2,2) = cosx
        makeYmat(2,3) = 0
        makeYmat(3,1) = 0
        makeYmat(3,2) = 0
        makeYmat(3,3) = 1
      end function

      subroutine calProdMat(tmat, nmat)
        implicit none
        double precision, intent(in) :: tmat(3,3)
        double precision, intent(inout) :: nmat(3,3)
        integer ii, jj, kk
        double precision tmp(3,3)
        do kk=1, 3
          do jj=1,3
            tmp(jj,kk) = 0.0
            do ii=1,3
              tmp(jj,kk) = tmp(jj,kk) + tmat(jj,ii)*nmat(ii,kk)
            end do
          end do
        end do
        do jj=1, 3
          do ii=1, 3
            nmat(ii,jj) = tmp(ii,jj)
          end do
        end do
      end subroutine
      subroutine calProdMatYT(ymat, tmat, nmat)
        implicit none
        double precision, intent(inout) :: nmat(3,3)
        double precision, intent(in) :: ymat(3,3), tmat(3,3)
        integer ii, jj, kk
        double precision tmp(3,3), tmp2(3,3)
        do kk=1, 3
          do jj=1,3
            tmp(jj,kk) = 0.0
            do ii=1,3
              tmp(jj,kk) = tmp(jj,kk) + tmat(jj,ii)*nmat(ii,kk)
            end do
          end do
        end do
        do kk=1, 3
          do jj=1,3
            tmp2(jj,kk) = 0.0
            do ii=1,3
              tmp2(jj,kk) = tmp2(jj,kk) + ymat(jj,ii)*tmp(ii,kk)
            end do
          end do
        end do
        do jj=1, 3
          do ii=1, 3
            nmat(ii,jj) = tmp2(ii,jj)
          end do
        end do
      end subroutine

      function PHp(gam, gam15, pi, m, a, eps, kap, zeta, k0)
        implicit none
        double precision, intent(in) :: gam, gam15, pi, m, a, eps
        double precision, intent(in) :: kap, zeta, k0
        double precision PHp
        PHp = (8.0*(pi**1.5)*gam15*eps*eps*a*a*a/gam)
     :       * (((zeta*a*k0)**2+a*a*m*m)**(-kap-1.5))
      end function

      function PHs(gam, gam15, pi, m, a, eps, kap, zeta, el0)
        implicit none
        double precision, intent(in) :: gam, gam15, pi, m, a, eps
        double precision, intent(in) :: kap, zeta, el0
        double precision PHs
        PHs = (8.0*(pi**1.5)*gam15*eps*eps*a*a*a/gam)
     :       * (((zeta*a*el0)**2+a*a*m*m)**(-kap-1.5))
      end function

      function PH(gam, gam15, pi, m, a, eps, kap, eta)
        implicit none
        double precision, intent(in) :: gam, gam15, pi, m, a, eps
        double precision, intent(in) :: kap, eta
        double precision PH
        PH = (8*(pi**1.5)*gam15*eps*eps*a*a*a/gam)
     :    * ((eta*eta+a*a*m*m)**(-kap-1.5))
      end function

      function XPPr(gamma0, theta, nu)
        implicit none
        double precision, intent(in) :: gamma0, theta, nu
        double precision XPPr, gam2, sin2
        gam2 = gamma0**2
        sin2 = (sin(theta))**2
        XPPr = nu*(-1.0+cos(theta)+2.0*sin2/gam2)-2.0
     :     + 4.0*sin2/gam2
        XPPr = XPPr / gam2
      end function

      function XPSt(gamma0, theta, nu)
        implicit none
        double precision, intent(in) :: gamma0, theta, nu
        double precision XPSt, cost
        cost = cos(theta)
        XPSt = nu*(1.0-2.0*cost/gamma0)-4.0*cost/gamma0
        XPSt = -sin(theta)*XPSt
      end function

      function XSPr(gamma0, theta, phi, nu)
        implicit none
        double precision, intent(in) :: gamma0, theta, phi, nu
        double precision XSPr, cost
        cost = cos(theta)
        XSPr = nu*(1.0-2.0*cost/gamma0)-4.0*cost/gamma0
        XSPr = XSPr*sin(theta)*cos(phi)/(gamma0**2)
      end function

      function XSSt(theta, phi, nu)
        implicit none
        double precision, intent(in) :: theta, phi, nu
        double precision XSSt, cos2t
        cos2t = cos(2.0*theta)
        XSSt = nu*(cos(theta)-cos2t)-2.0*cos2t
        XSSt = cos(phi)*XSSt
      end function

      function XSSp(theta, phi, nu)
        implicit none
        double precision, intent(in) :: theta, phi, nu
        double precision XSSp, cost
        cost = cos(theta)
        XSSp = sin(phi)*(nu*(cost-1.0)+2.0*cost)
      end function

      function XSSs1(theta, phi, nu)
        implicit none
        double precision, intent(in) :: theta, phi, nu
        double precision XSSs1, vXSSt, vXSSp
        vXSSt = XSSt(theta, phi, nu)
        vXSSp = XSSp(theta, phi, nu)
        XSSs1 = sqrt(vXSSt**2 + vXSSp**2)
      end function

      function gPPr(theta, el0, pi, gamma0, gam, gam15, a, eps
     :      , kap, eta, nu)
        implicit none
        double precision, intent(in) :: theta, el0, pi, gamma0
        double precision, intent(in) :: gam, gam15, a, eps, kap
        double precision, intent(in) :: eta, nu
        double precision gPPr, m, vXPPr
        vXPPr = XPPr(gamma0, theta, nu)
        m = (2.0*el0/gamma0)*sin(0.5*theta)
        gPPr = ((el0**4.0)/(4.0*pi))*(vXPPr**2)
     :        * PH(gam,gam15,pi,m,a,eps,kap,eta)
      end function

      function gPSt(theta, el0, pi, gamma0, gam, gam15, a, eps
     :      , kap, eta, nu)
        implicit none
        double precision, intent(in) :: theta, el0, pi, gamma0
        double precision, intent(in) :: gam, gam15, a, eps, kap
        double precision, intent(in) :: eta, nu
        double precision gPSt, m, vXPSt
        m = (el0/gamma0)*sqrt(1.0+gamma0**2-2.0*gamma0*cos(theta))
        vXPSt = XPSt(gamma0, theta, nu)
        gPSt = ((el0**4)/(4.0*pi*gamma0))*(vXPSt**2)
     :      *PH(gam,gam15,pi,m,a,eps,kap,eta)
      end function

      function gSPr(theta, phi, el0, pi, gamma0, gam, gam15, a, eps
     :      , kap, eta, nu)
        implicit none
        double precision, intent(in) :: theta, el0, pi, gamma0, phi
        double precision, intent(in) :: gam, gam15, a, eps, kap
        double precision, intent(in) :: eta, nu
        double precision gSPr, m, vXSPr
        m = (el0/gamma0)*sqrt(1.0+gamma0**2-2.0*gamma0*cos(theta))
        vXSPr = XSPr(gamma0, theta, phi, nu)
        gSPr = gamma0*((el0**4)/(4.0*pi))*(vXSPr**2)
     :      *PH(gam,gam15,pi,m,a,eps,kap,eta)
      end function

      function gSSs1(theta, phi, el0, pi, gamma0, gam, gam15, a, eps
     :     , kap, eta, nu)
        implicit none
        double precision, intent(in) :: theta, phi, el0, pi, gamma0
        double precision, intent(in) :: gam, gam15, a, eps, kap
        double precision, intent(in) :: eta, nu
        double precision gSSs1, m, vXSSs1
        m = 2.0*el0*sin(0.5*theta)
        vXSSs1 = XSSs1(theta, phi, nu)
        gSSs1 = ((el0**4)/(4.0*pi))*(vXSSs1**2)
     :     *PH(gam,gam15,pi,m,a,eps,kap,eta)
      end function

      function PDFHPPr(theta, el0, pi, gamma0, gam, gam15, a, eps
     :    , kap, eta, gPPr0, nu)
        implicit none
        double precision, intent(in) :: theta, el0, pi, gamma0, gam
        double precision, intent(in) :: gam15, a, eps, kap, eta
        double precision, intent(in) :: gPPr0, nu
        double precision PDFHPPr
        PDFHPPr = gPPr(theta,el0,pi,gamma0,gam,gam15,a,eps,kap
     :     ,eta,nu)*sin(theta)/(4.0*pi*gPPr0)
      end function
      function maxPDFHPPr(pi, gamma0, gam, gam15, a, eps, kap
     :     , eta, gPPr0, nu)
        implicit none
        double precision, intent(in) :: pi, gamma0, gam, gam15, a
        double precision, intent(in) :: eps, kap, eta, gPPr0
        double precision, intent(in) :: nu
        double precision maxPDFHPPr, th, dth, dtmp, mx
        integer nth
        nth = 1000
        dth = pi/dble(nth)
        maxPDFHPPr = 0.0d0
        do ii=1, nth
          th = (ii-1)*dth
          dtmp = PDFHPPr(th,el0,pi,gamma0,gam,gam15,a,eps,kap,eta
     :       ,gPPr0,nu)
          if(maxPDFHPPr .lt. dtmp) then
            maxPDFHPPr = dtmp
          end if
        end do
      end function

      function PDFHPSt(theta, el0, pi, gamma0, gam, gam15, a, eps
     :    , kap, eta, gPSt0, nu)
        implicit none
        double precision, intent(in) :: theta, el0, pi, gamma0, gam
        double precision, intent(in) :: gam15, a, eps, kap, eta
        double precision, intent(in) :: gPSt0, nu
        double precision PDFHPSt
        PDFHPSt = gPSt(theta,el0,pi,gamma0,gam,gam15,a,eps,kap
     :       ,eta,nu)*sin(theta)/(4.0*pi*gPSt0)
      end function
      function maxPDFHPSt(pi, gamma0, gam, gam15, a, eps, kap
     :     , eta, gPPr0, nu)
        implicit none
        double precision, intent(in) :: pi, gamma0, gam, gam15, a
        double precision, intent(in) :: eps, kap, eta, gPPr0
        double precision, intent(in) :: nu
        double precision maxPDFHPSt, th, dth, dtmp
        integer nth
        nth = 1000
        dth = pi/dble(nth)
        maxPDFHPSt = 0.0d0
        do ii=1, nth
          th = (ii-1)*dth
          dtmp = PDFHPSt(th,el0,pi,gamma0,gam,gam15,a,eps,kap,eta
     :       ,gPPr0,nu)
          if(maxPDFHPSt .lt. dtmp) then
            maxPDFHPSt = dtmp
          end if
        end do
      end function

      function PDFHSPr(theta, phi, el0, pi, gamma0, gam, gam15, a
     :    , eps, kap, eta, gSPr0, nu)
        implicit none
        double precision, intent(in) :: theta, phi, el0, pi, gamma0
        double precision, intent(in) :: gam, gam15, a, eps, kap, eta
        double precision, intent(in) :: gSPr0, nu
        double precision PDFHSPr
        PDFHSPr = gSPr(theta,phi,el0,pi,gamma0,gam,gam15,a,eps
     :       ,kap,eta,nu)*sin(theta)/(4.0*pi*gSPr0)
      end function
      function maxPDFHSPr(pi, gamma0, gam, gam15, a, eps, kap
     :     , eta, gPPr0, nu)
        implicit none
        double precision, intent(in) :: pi, gamma0, gam, gam15, a
        double precision, intent(in) :: eps, kap, eta, gPPr0
        double precision, intent(in) :: nu
        double precision maxPDFHSPr, th, dth, dtmp, phi, dphi
        integer nth, nphi
        nth = 1000
        nphi = 1000
        dth = pi/dble(nth)
        dphi = 2.0*pi/dble(nphi)
        maxPDFHSPr = 0.0d0
        do ii=1, nth
          th = (ii-1)*dth
          do jj=1, nphi
            phi = (jj-1)*dphi
            dtmp = PDFHSPr(th,phi,el0,pi,gamma0,gam,gam15,a,eps,kap
     :       ,eta,gPPr0,nu)
            if(maxPDFHSPr .lt. dtmp) then
              maxPDFHSPr = dtmp
            end if
          end do
        end do
      end function

      function PDFHSSs1(theta, phi, el0, pi, gamma0, gam, gam15, a
     :    , eps, kap, eta, gSSs10, nu)
        implicit none
        double precision, intent(in) :: theta, phi, el0, pi, gamma0
        double precision, intent(in) :: gam, gam15, a, eps, kap, eta
        double precision, intent(in) :: gSSs10, nu
        double precision PDFHSSs1
        PDFHSSs1 = gSSs1(theta,phi,el0,pi,gamma0,gam,gam15,a,eps,
     :      kap,eta,nu)*sin(theta)/(4.0*pi*gSSs10)
      end function
      function maxPDFHSSs1(pi, gamma0, gam, gam15, a, eps, kap
     :     , eta, gPPr0, nu)
        implicit none
        double precision, intent(in) :: pi, gamma0, gam, gam15, a
        double precision, intent(in) :: eps, kap, eta, gPPr0
        double precision, intent(in) :: nu
        double precision maxPDFHSSs1, th, dth, dtmp, phi, dphi
        integer nth, nphi
        nth = 1000
        nphi = 1000
        dth = pi/dble(nth)
        dphi = 2.0*pi/dble(nphi)
        maxPDFHSSs1 = 0.0d0
        do ii=1, nth
          th = (ii-1)*dth
          do jj=1, nphi
            phi = (jj-1)*dphi
            dtmp = PDFHSSs1(th,phi,el0,pi,gamma0,gam,gam15,a,eps,kap
     :       ,eta,gPPr0,nu)
            if(maxPDFHSSs1 .lt. dtmp) then
              maxPDFHSSs1 = dtmp
            end if
          end do
        end do
      end function

      function fgPPr0(el0, pi, gamma0, gam, gam15, a, eps, kap,
     :         eta, nu)
        implicit none
        double precision a, eps, kap, eta, pi, gam, gam15
        double precision fgPPr0, el0, gamma0, nu
        double precision dth, th
        integer nth, ii
        dth = 0.001*pi/180.0
        nth = int(pi/dth)
        fgPPr0 = 0.5*gPPr(dth*(nth-1),el0,pi,gamma0,gam,gam15,
     :           a,eps,kap,eta,nu)*sin(dth*(nth-1))
        do ii=2, nth-1
          th = (ii-1)*dth
          fgPPr0 = fgPPr0
     :     +gPPr(th,el0,pi,gamma0,gam,gam15,a,eps,kap,eta,nu)
     :     *sin(th)
        end do
        fgPPr0 = fgPPr0*0.5*dth
      end function

      function fgPSt0(el0, pi, gamma0, gam, gam15, a, eps, kap,
     :         eta, nu)
        implicit none
        double precision a, eps, kap, eta, pi, gam, gam15
        double precision fgPSt0, el0, gamma0, nu
        double precision dth, th
        integer nth, ii
        dth = 0.001*pi/180.0
        nth = int(pi/dth)
        fgPSt0 = 0.5*gPSt(dth*(nth-1),el0,pi,gamma0,gam,gam15,
     :           a,eps,kap,eta,nu)*sin(dth*(nth-1))
        do ii=2, nth-1
          th = (ii-1)*dth
          fgPSt0 = fgPSt0
     :     +gPSt(th,el0,pi,gamma0,gam,gam15,a,eps,kap,eta,nu)
     :     *sin(th)
        end do
        fgPSt0 = fgPSt0*0.5*dth
      end function

      function fgSPr0(el0, pi, gamma0, gam, gam15, a, eps, kap,
     :         eta, nu)
        implicit none
        double precision a, eps, kap, eta, pi, gam, gam15
        double precision fgSPr0, el0, gamma0, nu
        double precision dth, th, dphi, phi
        integer nth, ii, nphi, jj
        parameter(nth = 2000)
        parameter(nphi = 2000)
        double precision arphi(nphi)
        dth = pi/dble(nth)
        dphi = 2.0*pi/dble(nphi)
        do ii=1, nphi
          phi = (ii-1)*dphi
          th = dth*(nth-1)
          arphi(ii) = 0.5*gSPr(th,phi,el0,pi,gamma0,gam,gam15,
     :           a,eps,kap,eta,nu)*sin(th)
          do jj=2, nth-1
            th = (jj-1)*dth
            arphi(ii) = arphi(ii)
     :       +gSPr(th,phi,el0,pi,gamma0,gam,gam15,a,eps,kap,
     :           eta,nu)*sin(th)
          end do
          arphi(ii) = arphi(ii)*dth
        end do
        fgSPr0 = 0.5*(arphi(1) + arphi(nphi))
        do ii=2, nth-1
          fgSPr0 = fgSPr0 + arphi(ii)
        end do
        fgSPr0 = fgSPr0*dphi/(4.0*pi)
      end function

      function fgSSs10(el0, pi, gamma0, gam, gam15, a, eps, kap,
     :         eta, nu)
        implicit none
        double precision a, eps, kap, eta, pi, gam, gam15
        double precision fgSSs10, el0, gamma0, nu
        double precision dth, th, dphi, phi
        integer nth, ii, nphi, jj
        parameter(nth = 2000)
        parameter(nphi = 2000)
        double precision arphi(nphi)
        dth = pi/dble(nth)
        dphi = 2.0*pi/dble(nphi)
        do ii=1, nphi
          phi = (ii-1)*dphi
          th = dth*(nth-1)
          arphi(ii) = 0.5*gSSs1(th,phi,el0,pi,gamma0,gam,gam15,
     :           a,eps,kap,eta,nu)*sin(th)
          do jj=2, nth-1
            th = (jj-1)*dth
            arphi(ii) = arphi(ii)
     :       +gSSs1(th,phi,el0,pi,gamma0,gam,gam15,a,eps,kap,
     :           eta,nu)*sin(th)
          end do
          arphi(ii) = arphi(ii)*dth
        end do
        fgSSs10 = 0.5*(arphi(1) + arphi(nphi))
        do ii=2, nth-1
          fgSSs10 = fgSSs10 + arphi(ii)
        end do
        fgSSs10 = fgSSs10*dphi/(4.0*pi)
      end function


      subroutine rHPP(el0, pi, gamma0, gam, gam15, a
     :      , eps, kap, eta, gPP0, t, f, nu, hHPP)
        implicit none
        double precision, intent(in) :: el0, pi
        double precision, intent(in) :: gamma0, gam, gam15, a, nu
        double precision, intent(in) :: eps, kap, eta, gPP0
        double precision, intent(out) :: t, f
        double precision z, hHPP, dtmp, t1, f1
        double precision grnd
        external grnd
        dtmp = PDFHPPr(0.0d0,el0,pi,gamma0,gam,gam15,a,eps,kap,eta
     :     ,gPP0, nu)
        z = hHPP
        do while(z .gt. dtmp)
          t1 = pi*grnd()
          f1 = 2.0*pi*grnd()
          z = hHpp * grnd()
          dtmp = PDFHPPr(t1,el0,pi,gamma0,gam,gam15,a,eps,kap,eta
     :     ,gPP0, nu)
        end do
        t = t1
        f = f1
      end subroutine
      subroutine rHPS(el0, pi, gamma0, gam, gam15, a
     :      , eps, kap, eta, gPS0, t, f, nu, hHPS)
        implicit none
        double precision, intent(in) :: el0, pi
        double precision, intent(in) :: gamma0, gam, gam15, a, nu
        double precision, intent(in) :: eps, kap, eta, gPS0
        double precision, intent(out) :: t, f
        double precision z, hHPS, dtmp, t1, f1
        double precision grnd
        external grnd
        dtmp = PDFHPSt(0.0d0,el0,pi,gamma0,gam,gam15,a,eps,kap,eta
     :     ,gPS0, nu)
        z = hHPS
        do while(z .gt. dtmp)
          t1 = pi*grnd()
          f1 = 2.0*pi*grnd()
          z = hHPS * grnd()
          dtmp = PDFHPSt(t1,el0,pi,gamma0,gam,gam15,a,eps,kap,eta
     :     ,gPS0, nu)
        end do
        t = t1
        f = f1
      end subroutine
      subroutine rHSP(el0, pi, gamma0, gam, gam15, a
     :      , eps, kap, eta, gSP0, t, f, nu, hHSP)
        implicit none
        double precision, intent(in) :: el0, pi
        double precision, intent(in) :: gamma0, gam, gam15, a, nu
        double precision, intent(in) :: eps, kap, eta, gSP0
        double precision, intent(out) :: t, f
        double precision z, hHSP, dtmp, t1, f1
        double precision grnd
        external grnd
        dtmp = PDFHSPr(0.0d0,0.0d0,el0,pi,gamma0,gam,gam15,a,eps,kap,
     :     eta,gSP0, nu)
        z = hHSP
        do while(z .gt. dtmp)
          t1 = pi*grnd()
          f1 = 2.0*pi*grnd()
          z = hHSP * grnd()
          dtmp = PDFHSPr(t1,f1,el0,pi,gamma0,gam,gam15,a,eps,kap,eta
     :     ,gSP0, nu)
        end do
        t = t1
        f = f1
      end subroutine
      subroutine rHSS(el0, pi, gamma0, gam, gam15, a
     :      , eps, kap, eta, gSS0, t, f, nu, hHSS)
        implicit none
        double precision, intent(in) :: el0, pi
        double precision, intent(in) :: gamma0, gam, gam15, a, nu
        double precision, intent(in) :: eps, kap, eta, gSS0
        double precision, intent(out) :: t, f
        double precision z, hHSS, dtmp, t1, f1
        double precision grnd
        external grnd
        dtmp = PDFHSSs1(0.0d0,0.0d0,el0,pi,gamma0,gam,gam15,a,eps,kap,
     :     eta,gSS0, nu)
        z = hHSS
        do while(z .gt. dtmp)
          t1 = pi*grnd()
          f1 = 2.0*pi*grnd()
          z = hHSS * grnd()
          dtmp = PDFHSSs1(t1,f1,el0,pi,gamma0,gam,gam15,a,eps,kap,
     :     eta,gSS0, nu)
        end do
        t = t1
        f = f1
      end subroutine



      function fCLp(gam, gam05, kap, a, zeta, k0, pi)
        implicit none
        double precision, intent(in) :: gam, gam05, kap
        double precision, intent(in) :: a, zeta, k0, pi
        double precision fCLp
        if(kap .eq. 0.5) then
          fCLp = log(zeta*a*k0)
        else
          fCLp = sqrt(pi)*gam05*(1.0-(zeta*a*k0)**(-2*kap+1))
     :          /(gam*(2*kap-1))
        end if
      end function

      function fCLs(gam, gam05, kap, a, zeta, el0, pi)
        implicit none
        double precision, intent(in) :: gam, gam05, kap
        double precision, intent(in) :: a, zeta, el0, pi
        double precision fCLs
        if(kap .eq. 0.5) then
          fCLs = log(zeta*a*el0)
        else
          fCLs = sqrt(pi)*gam05*(1.0-(zeta*a*el0)**(-2*kap+1))
     :          /(gam*(2*kap-1))
        end if
      end function

      function fCL(gam, gam05, kap, eta, pi)
        implicit none
        double precision, intent(in) :: gam, gam05, kap
        double precision, intent(in) :: eta, pi
        double precision fCL
        if(kap .eq. 0.5) then
          fCL = log(eta)
        else
          fCL = sqrt(pi)*gam05*(1-(eta**(-2*kap+1)))/(gam*(2*kap-1))
        end if
      end function


      subroutine rLP(pi, tq, fq, sigmaLP)
        implicit none
        double precision, intent(in) :: pi, sigmaLP
        double precision, intent(out) :: tq, fq
        double precision grnd
        double precision grndtmp
        external grnd
        fq = 2.0*pi*grnd()
        grndtmp = grnd()
        if(grndtmp.eq.1) then
          grndtmp = grnd()
        end if
        tq = frLPt(sigmaLP, grndtmp)
      end subroutine

      function frLPt(sigmaLP, p)
        implicit none
        double precision, intent(in) :: sigmaLP, p
        double precision frLPt
        frLPt = sqrt(-2.0*(sigmaLP**2)*log(1.0-p))
      end function

      subroutine rLS(pi, tq, fq, sigmaLS)
        implicit none
        double precision, intent(in) :: pi, sigmaLS
        double precision, intent(out) :: tq, fq
        double precision grnd, p
        external grnd
        fq = 2.0*pi*grnd()
        p = grnd()
        if(p .eq. 1) then
          p = grnd()
        end if
        tq = frLSt(sigmaLS, p)
        
      end subroutine

      function frLSt(sigmaLS, p)
        implicit none
        double precision, intent(in) :: sigmaLS, p
        double precision frLSt
        frLSt = sqrt(-2.0*(sigmaLS**2)*log(1.0-p))
      end function

      function PDFRp(theta, phi, pi)
        implicit none
        double precision, intent(in) :: theta, phi, pi
        double precision PDFRp
        PDFRp = (15.0/(16.0*pi))*(sin(theta)**5)*(sin(2.0*phi)**2)
      end function
      function maxPDFRp(pi)
        implicit none
        double precision, intent(in) :: pi
        double precision maxPDFRp, th, dth, dtmp, phi, dphi
        integer nth, nphi
        nth = 1000
        nphi = 1000
        dth = pi/dble(nth)
        dphi = 2.0*pi/dble(nphi)
        maxPDFRp = 0.0d0
        do ii=1, nth
          th = (ii-1)*dth
          do jj=1, nphi
            phi = (jj-1)*dphi
            dtmp = PDFRp(th,phi,pi)
            if(maxPDFRp .lt. dtmp) then
              maxPDFRp = dtmp
            end if
          end do
        end do
      end function

      function PDFRs(theta, phi, pi)
        implicit none
        double precision, intent(in) :: theta, phi, pi
        double precision PDFRs
        PDFRs = 5.0*(sin(2*theta)**2)*(sin(2*phi)**2)/8.0
     :      + 5.0*(sin(theta)**2)*(cos(2*phi)**2)/2.0
        PDFRs = PDFRs*sin(theta)/(4.0*pi)
      end function
      function maxPDFRs(pi)
        implicit none
        double precision, intent(in) :: pi
        double precision maxPDFRs, th, dth, dtmp, phi, dphi
        integer nth, nphi
        nth = 1000
        nphi = 1000
        dth = pi/dble(nth)
        dphi = 2.0*pi/dble(nphi)
        maxPDFRs = 0.0d0
        do ii=1, nth
          th = (ii-1)*dth
          do jj=1, nphi
            phi = (jj-1)*dphi
            dtmp = PDFRs(th,phi,pi)
            if(maxPDFRs .lt. dtmp) then
              maxPDFRs = dtmp
            end if
          end do
        end do
      end function

      subroutine frRp(pi, t, f, hRp)
        implicit none
        double precision, intent(in) :: pi
        double precision, intent(out) :: t, f
        double precision z, hRp, dtmp, t1, f1
        double precision grnd
        external grnd
        dtmp = PDFRp(0.0d0,0.0d0, pi)
        z = hRp
        do while(z .gt. dtmp)
          t1 = pi*grnd()
          f1 = 2.0*pi*grnd()
          z = hRp * grnd()
          dtmp = PDFRp(t1,f1,pi)
        end do
        t = t1
        f = f1
      end subroutine

      subroutine frRs(pi, t, f, hRs)
        implicit none
        double precision, intent(in) :: pi
        double precision, intent(out) :: t, f
        double precision z, hRs, dtmp, t1, f1
        double precision grnd
        external grnd
        dtmp = PDFRs(0.0d0,0.0d0, pi)
        z = hRs
        do while(z .gt. dtmp)
          t1 = pi*grnd()
          f1 = 2.0*pi*grnd()
          z = hRs * grnd()
          dtmp = PDFRs(t1,f1,pi)
        end do
        t = t1
        f = f1
      end subroutine

      function randnorm(pi, mean, std)
        implicit none
        double precision, intent(in) :: pi
        double precision, intent(in) :: std
        double precision, intent(in) :: mean
        double precision grnd, r1, r2
        double precision randnorm
        external grnd
        r1 = grnd()
        r2 = grnd()
        randnorm = sqrt(-2*log(r1))*sin(2*pi*r2)*std+mean
      end function



      function checkReachStation(xyz, stposmin, stposmax, nofst)
        implicit none
        double precision stposmin(:,:), stposmax(:,:)
        double precision xyz(:)
        integer checkReachStation, ii, jj, nofst
        logical flag
        flag = .false.
        checkReachStation = -1
        do ii=1, nofst
          do jj=1, 3
            if((xyz(jj).lt.stposmin(ii,jj)).or.
     :        (xyz(jj).gt.stposmax(ii,jj))) then
              flag = .false.
              exit
            end if
            flag = .true.
          end do
          if(flag) then
            checkReachStation = ii
            exit
          end if
        end do
      end function

      function atan3(y, x, pi)
        implicit none
        double precision, intent(in) :: x, y, pi
        double precision atan3
        atan3 = atan2(y, x)
        if(atan3 .lt. 0) then
          atan3 = atan3 + 2.0*pi
        end if
      end function

      function atanRad(theta, phi, pi)
        implicit none
        double precision, intent(in) :: theta, phi, pi
        double precision atanRad, bt, bp
        bt = sqrt(5.0d0/2.0d0)*0.5*sin(2*theta)*sin(2*phi)
        bp = sqrt(5.0d0/2.0d0)*sin(theta)*cos(2*phi)
        atanRad = atan2(bp, bt)
        if(atanRad .lt. 0) then
          atanRad = atanRad + 2.0*pi
        end if
      end function

      subroutine makeEnv(env, nofshot, nofst, dxpos, nt)
        implicit none
        double precision env(:,:,:)
        double precision dxpos
        double precision v, fac
        integer nofshot, nofst, ii, jj, kk, nt
        do ii=1, nofst
          v = dxpos**3
          fac = 1.0/(v*nofshot)
          do jj=1, 3
            do kk=1, nt
              env(ii,jj,kk) = sqrt(env(ii,jj,kk)*fac)
c              env(ii,jj,kk) = env(ii,jj,kk)*fac
            end do
          end do
        end do
      end subroutine

      subroutine fileOutGrs(outdir, env, nofst, dt, nt)
        implicit none
        character outdir*70, filename*80
        double precision env(:,:,:)
        double precision dt
        integer ii, jj, nofst, itmp, nt
        itmp = len_trim(outdir)
        write(6,*) 'nt', nt
        do ii=1, nofst
          do jj=1, 3
            if(ii .le. 10) then
		            filename = outdir(1:itmp)//'grs_'//format3(ii-1)//
     :             '_'//format1(jj-1)//'.txt'
            else
		            filename = outdir(1:itmp)//'grs_'//format3(ii-1)//
     :             '_'//format1(jj-1)//'.txt'
            end if
            open(20, file=filename)
            do kk=1, nt
              write(20,*) ((kk-1)*dt), env(ii,jj,kk)
            end do
            close(20)
          end do
        end do
      end subroutine

      function format1(n)
        implicit none
        integer n
        character format1*1
        write(format1,'(i1)') n
      end function
      function format2(n)
        implicit none
        integer n
        character format2*2
        write(format2,'(i2)') n
      end function
      function format3(n)
        integer n
        character format3*3
        if(n .lt. 10) then
           write(format3,'(a2,i1)') '00', n
        else if(n .lt. 100) then
           write(format3,'(a1,i2)') '0', n
        else if(n .lt. 1000) then
           write(format3,'(i3)') n
        end if
      end function



cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C  (C) Copr. 1986-92 Numerical Recipes Software
      FUNCTION gammln(xx)
        implicit none
      double precision gammln,xx
      INTEGER j
      DOUBLE PRECISION ser,stp,tmp,x,y,cof(6)
      SAVE cof,stp
      DATA cof,stp/76.18009172947146d0,-86.50532032941677d0,
     *24.01409824083091d0,-1.231739572450155d0,.1208650973866179d-2,
     *-.5395239384953d-5,2.5066282746310005d0/
      x=xx
      y=x
      tmp=x+5.5d0
      tmp=(x+0.5d0)*log(tmp)-tmp
      ser=1.000000000190015d0
      do 11 j=1,6
        y=y+1.d0
        ser=ser+cof(j)/y
11    continue
      gammln=tmp+log(stp*ser/x)
      return
      END FUNCTION


      END
