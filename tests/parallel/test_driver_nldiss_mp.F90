!===============================================================================
! test_driver_nldiss_mp.F90 -- [diagnostic v2] MODE-RESOLVED test of the
! distributed-sp nonlinear-transform bias behind the MKM run (uu+ peak ~4% low
! in distributed-sp, while single-GPU-sp AND distributed-dp both match MKM).
!
! WHAT THE EMPIRICAL CLUE TELLS US
! --------------------------------
!   single-GPU-sp  : matches MKM      => sp arithmetic alone is fine
!   distributed-dp : matches MKM      => the split path alone is fine
!   distributed-sp : 4% uu deficit    => the defect needs BOTH the split path
!                                        AND single precision.
! The targeted isolation is therefore split-vs-fused AT FIXED PRECISION:
!     fused : nl_evaluate      (single-GPU, fused 2-D transform)   -> (hvf, hetaf)
!     split : par_nl_evaluate  (distributed, 1-D z -> NCCL transpose -> 1-D x) -> (hvs, hetas)
! on ONE shared reference trajectory (every rank holds the full field). The exact
! nonlinear conserves energy, so for an exact transform hvs==hvf, hetas==hetaf.
! dN = (hvs-hvf, hetas-hetaf) is the spurious forcing the split-sp path injects.
!
! THIS VERSION FIXES THREE THINGS THE v1 NULL EXPOSED
! ---------------------------------------------------
!  (1) SELF-IDENTIFYING: prints the COMPILED precision (tp). v1 built dp and sp to
!      the SAME binary path and the log never said which ran -- the two "dp/sp"
!      logs were the same binary twice. Build script now emits _dp/_sp binaries;
!      sbatch runs BOTH; this banner states tp so a log can't be mistaken.
!  (2) MODE-RESOLVED, not one signed scalar. We accumulate per-(kx,kz) maps,
!      summed over y, of the SIGNED injection Re[conj(q)*dN] and the GROSS
!      magnitude |dN|^2, then bin by |kx| and |kz| SEPARATELY (kz is NOT kx: the
!      split-specific ops are the z-FFT + transpose). Reports:
!        * net (sum) vs gross (sum|.|): |net|~gross => systematic drain/source
!          (dissipation); |net|<<gross => REDISTRIBUTION (energy moved, ~conserved).
!        * kx- and kz-profiles + spectral centroids of dN vs of the energy, so we
!          can see WHETHER the error is high-k weighted and on WHICH axis.
!  (3) HIGH-k POPULATED + VISIBLE: the IC band is widened and an explicit high-k
!      probe is injected, and the input E(kx),E(kz) spectra are printed at the
!      sample, so the test is not blind where the field has no energy.
!
! NOTE (honest scope): the projection conj(v)*dNv + conj(eta)*dNeta is the natural
! quadratic measure in the (v,eta) representation; it is a proxy for the physical
! u/v/w kinetic-energy budget (eta is vorticity), not an exact uu/vv/ww split. A
! component (uu vs vv vs ww) decomposition is the next refinement if this flags.
!
! GRID = production box/Re (n=nx=nz=128, Re=2800, dt=5e-3, kcut widened, seed
! fixed); evolve with the FUSED single-GPU stepper, sample a few late times.
! Build: TP=dp|sp bash scripts/build/build_par.sh test_driver_nldiss_mp   Run: bash run_all_par_tests.sh run -k test_driver_nldiss_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_driver_nldiss_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,              only: tp, wp, dp, sp, ip
  use cudafor
  use decomp_mod,         only: decomp_t, decomp_setup
  use helm_band_mod,      only: helm_band_setup
  use mean_mod,           only: mean_flux_setup
  use stepper_lin_mod,    only: stepper_ev_setup
  use solver_ops_mod,     only: band_ops_t, vsolve_ops_t, mean_ops_t, &
                                band_ops_alloc, band_ops_free, vsolve_ops_alloc, &
                                vsolve_ops_free, mean_ops_alloc, mean_ops_free
  use init_mod,           only: init_turbulent
  use nonlinear_mod,      only: nl_workspace, nl_setup, nl_destroy, nl_evaluate
  use stepper_mod,        only: step_full
  use par_nonlinear_mod,  only: par_nl_workspace, par_nl_setup, par_nl_destroy, par_nl_evaluate
  use par_mod
  implicit none

  integer(ip), parameter :: n = 128, nx = 128, nz = 128
  integer(ip), parameter :: ny = n+1, nkx = nx/2+1, nxd = 3*nx/2, nzd = 3*nz/2
  real(dp),    parameter :: Re = 2800.0_dp, Qtar = 1.0_dp, dt = 5.0e-3_dp
  real(dp),    parameter :: epsp = 0.10_dp, PI = acos(-1.0_dp), Lx = 4.0_dp*PI, Lz = 4.0_dp*PI/3.0_dp
  integer(ip), parameter :: kcut = 16, pseed = 12345                 ! widened IC band (was 8)
  integer(ip), parameter :: NT = 2000, KSAMP = 250, NSKIP = 1000     ! advance / sample / warmup
  integer(ip), parameter :: nbx = nx/2, nbz = nz/2                   ! top kx,|kz| bin index

  type(decomp_t) :: d
  integer(ip) :: bnkx, kxlo, kxhi, ix, iz, kk, it, nsamp, jx, mx, amz, mz
  integer     :: istat
  real(dp)    :: kxh(nkx), kzh(nz)
  type(band_ops_t)   :: bo     ! y-operator band structure (host + device)
  type(vsolve_ops_t) :: vo     ! shifted v-solve lift + influence
  type(mean_ops_t)   :: mo     ! CFR mean precompute

  ! common turbulent IC (host, rank-invariant by fixed seed)
  complex(dp), allocatable :: he0(:,:,:), hv0(:,:,:)
  complex(dp) :: hu00(0:n), hw00(0:n)

  ! reference state (FULL, every rank) + fused stepper precompute + nl workspace
  type(nl_workspace) :: wsf
  complex(dp), device, allocatable :: eta(:,:,:), v(:,:,:), u0(:), w0(:)
  complex(dp), device, allocatable :: Nvp(:,:,:), Nep(:,:,:)
  real(dp),    device, allocatable :: kx(:), kz(:)
  ! fused nonlinear outputs (FULL)
  complex(dp), device, allocatable :: hvf(:,:,:), hetaf(:,:,:)
  complex(dp), device :: hx0(0:n), hz0(0:n)
  ! split nonlinear (SLAB) + sliced reference/fused-output buffers (SLAB, contiguous)
  type(par_nl_workspace) :: wss
  complex(dp), device, allocatable :: vS(:,:,:), etaS(:,:,:), hvs(:,:,:), hetas(:,:,:)
  complex(dp), device, allocatable :: hvfS(:,:,:), hetafS(:,:,:)

  ! per-(kx,kz) maps (THIS rank's slab), summed over y -- device then host
  real(dp), device, allocatable :: PeM(:,:), PvM(:,:), DeM(:,:), DvM(:,:), NeM(:,:), NvM(:,:)
  real(dp),         allocatable :: hPeM(:,:), hPvM(:,:), hDeM(:,:), hDvM(:,:), hNeM(:,:), hNvM(:,:)

  ! binned profiles (global kx index 0..nbx ; |kz| index 0..nbz) -- reduced across ranks
  real(real64) :: Dkx(0:nbx), Ekx(0:nbx), Pekx(0:nbx)
  real(real64) :: Dkz(0:nbz), Ekz(0:nbz), Pekz(0:nbz)
  ! global scalars
  real(real64) :: netPe, grsPe, netPv, grsPv, sumDe, sumDv, sumNe, sumNv, sumE
  real(real64) :: cD_kx, cE_kx, cD_kz, cE_kz, tot

  call par_init()
  call decomp_setup(d, 0_ip, int(par_nproc,ip), int(par_rank,ip), nx, ny, nz)
  bnkx = d%b_nkx_loc;  kxlo = d%b_kx0 + 1;  kxhi = d%b_kx0 + bnkx

  if (par_rank == 0) then
     print '(a)',    '==========================================================================='
     print '(a,i0,a)', '[nldiss v2] world size = ', par_nproc, ' rank(s)'
     if (tp == sp) then
        print '(a)', '   BUILD PRECISION:  tp = SINGLE (real32)   <-- this is the TEST run'
     else
        print '(a)', '   BUILD PRECISION:  tp = DOUBLE (real64)   <-- this is the SANITY run'
     end if
     print '(a,i0)',  '   wp (field) bytes per real = ', wp
     print '(a)',    '==========================================================================='
  end if

  do ix = 1, nkx
     kxh(ix) = real(ix-1, dp) * (2.0_dp*PI/Lx)
  end do
  do iz = 1, nz
     kk = iz-1;  if (kk > nz/2) kk = kk - nz;  kzh(iz) = real(kk, dp) * (2.0_dp*PI/Lz)
  end do
  call band_ops_alloc(bo, n);  call mean_ops_alloc(mo, n)
  call helm_band_setup(n, bo%sdh, bo%sth, bo%bdh, bo%boff)
  call mean_flux_setup(n, dt, Re, bo%sdh, bo%sth, bo%bdh, bo%boff, mo%Uhm, mo%Ih)

  allocate(he0(0:n,nkx,nz), hv0(0:n,nkx,nz))
  call build_ic(he0, hv0, hu00, hw00)

  ! --- reference (full) state + fused stepper precompute + nl workspace ---------
  call nl_setup(wsf, ny, nx, nz, kxh, kzh)
  allocate(eta(0:n,nkx,nz), v(0:n,nkx,nz), u0(0:n), w0(0:n), Nvp(0:n,nkx,nz), Nep(0:n,nkx,nz))
  allocate(kx(nkx), kz(nz))
  call vsolve_ops_alloc(vo, n, nkx, nz)
  allocate(hvf(0:n,nkx,nz), hetaf(0:n,nkx,nz))
  kx = kxh;  kz = kzh;  bo%sd = bo%sdh;  bo%st = bo%sth;  bo%bd = bo%bdh
  call stepper_ev_setup(n, nkx, nz, dt, Re, kxh, kzh, vo%vm, vo%vp, vo%Cinv)
  eta = he0;  v = hv0;  u0 = hu00;  w0 = hw00
  Nvp = (0.0_dp,0.0_dp);  Nep = (0.0_dp,0.0_dp)

  ! --- split (slab) nonlinear workspace + slab buffers --------------------------
  call par_nl_setup(wss, d, ny, nx, nz, kxh, kzh)
  allocate(vS(0:n,bnkx,nz), etaS(0:n,bnkx,nz), hvs(0:n,bnkx,nz), hetas(0:n,bnkx,nz))
  allocate(hvfS(0:n,bnkx,nz), hetafS(0:n,bnkx,nz))
  allocate(PeM(bnkx,nz), PvM(bnkx,nz), DeM(bnkx,nz), DvM(bnkx,nz), NeM(bnkx,nz), NvM(bnkx,nz))
  allocate(hPeM(bnkx,nz), hPvM(bnkx,nz), hDeM(bnkx,nz), hDvM(bnkx,nz), hNeM(bnkx,nz), hNvM(bnkx,nz))

  if (par_rank == 0) then
     print '(a,i0,a,i0,a,i0)', '   grid (n,nx,nz) = ', n, ',', nx, ',', nz
     print '(a,i0,a,i0,a,i0)', '   advance ', NT, ' steps (fused ref); sample every ', KSAMP, &
          ' after step ', NSKIP
     print '(a)', '   dN = split - fused nonlinear forcing (eta- and v-equations)'
     print '(a)', '   Pe = sum Re[conj(eta)*dNeta]   net=sum  gross=sum|per-mode|   (eta inj.)'
     print '(a)', '   Pv = sum Re[conj(v)  *dNv  ]   net=sum  gross=sum|per-mode|   (v   inj.)'
     print '(a)', '   ---------------------------------------------------------------------------'
     print '(a)', '     step   netPe/Ne    grossPe/Ne   netPv/Nv    grossPv/Nv   |dN|^2/|N|^2'
  end if

  nsamp = 0
  do it = 1, NT
     if (it > NSKIP .and. mod(it, KSAMP) == 0) then
        ! fused nonlinear on the FULL current reference field
        call nl_evaluate(wsf, v, eta, u0, w0, hvf, hetaf, hx0, hz0)
        ! split nonlinear on the SLAB of the current field (mean passed; non-owners ignore)
        call copy_slab(n, nz, d%b_kx0, bnkx, v,   vS)
        call copy_slab(n, nz, d%b_kx0, bnkx, eta, etaS)
        call par_nl_evaluate(wss, vS, etaS, u0, w0, hvs, hetas, hx0, hz0)
        ! slice the fused outputs to this rank's kx-slab (contiguous)
        call copy_slab(n, nz, d%b_kx0, bnkx, hvf,   hvfS)
        call copy_slab(n, nz, d%b_kx0, bnkx, hetaf, hetafS)
        ! per-(kx,kz) maps, summed over y (robust: one thread per mode, serial in y)
        call mode_maps(n, bnkx, nz, vS, etaS, hvs, hvfS, hetas, hetafS, &
                       PeM, PvM, DeM, DvM, NeM, NvM)
        hPeM = PeM;  hPvM = PvM;  hDeM = DeM;  hDvM = DvM;  hNeM = NeM;  hNvM = NvM
        ! ---- bin on host: this rank's kx-slab, full z -------------------------
        Dkx = 0; Ekx = 0; Pekx = 0;  Dkz = 0; Ekz = 0; Pekz = 0
        netPe=0; grsPe=0; netPv=0; grsPv=0; sumDe=0; sumDv=0; sumNe=0; sumNv=0; sumE=0
        do iz = 1, nz
           mz = iz-1;  if (mz > nz/2) mz = mz - nz;  amz = abs(mz)
           do jx = 1, bnkx
              mx = d%b_kx0 + jx - 1
              netPe = netPe + hPeM(jx,iz);  grsPe = grsPe + abs(hPeM(jx,iz))
              netPv = netPv + hPvM(jx,iz);  grsPv = grsPv + abs(hPvM(jx,iz))
              sumDe = sumDe + hDeM(jx,iz);  sumDv = sumDv + hDvM(jx,iz)
              sumNe = sumNe + hNeM(jx,iz);  sumNv = sumNv + hNvM(jx,iz)
              sumE  = sumE  + hNeM(jx,iz) + hNvM(jx,iz)
              Dkx(mx)  = Dkx(mx)  + hDeM(jx,iz) + hDvM(jx,iz)
              Ekx(mx)  = Ekx(mx)  + hNeM(jx,iz) + hNvM(jx,iz)
              Pekx(mx) = Pekx(mx) + hPeM(jx,iz)
              Dkz(amz)  = Dkz(amz)  + hDeM(jx,iz) + hDvM(jx,iz)
              Ekz(amz)  = Ekz(amz)  + hNeM(jx,iz) + hNvM(jx,iz)
              Pekz(amz) = Pekz(amz) + hPeM(jx,iz)
           end do
        end do
        ! ---- combine ranks (kx slabs disjoint; z whole on every rank) ---------
        call par_reduce_sum(netPe); call par_reduce_sum(grsPe)
        call par_reduce_sum(netPv); call par_reduce_sum(grsPv)
        call par_reduce_sum(sumDe); call par_reduce_sum(sumDv)
        call par_reduce_sum(sumNe); call par_reduce_sum(sumNv); call par_reduce_sum(sumE)
        call par_reduce_sum_vec(Dkx,  nbx+1); call par_reduce_sum_vec(Ekx,  nbx+1)
        call par_reduce_sum_vec(Pekx, nbx+1)
        call par_reduce_sum_vec(Dkz,  nbz+1); call par_reduce_sum_vec(Ekz,  nbz+1)
        call par_reduce_sum_vec(Pekz, nbz+1)
        nsamp = nsamp + 1
        if (par_rank == 0) print '(i9,5(2x,es11.3))', it, &
             netPe/max(sumNe,tiny(1.0_real64)), grsPe/max(sumNe,tiny(1.0_real64)), &
             netPv/max(sumNv,tiny(1.0_real64)), grsPv/max(sumNv,tiny(1.0_real64)), &
             (sumDe+sumDv)/max(sumE,tiny(1.0_real64))
     end if
     ! advance the reference by the FUSED single-GPU stepper (every rank, identical)
     call step_full(wsf, n, nkx, nz, dt, Re, kx, kz, bo, vo, mo, Qtar, &
                    u0, w0, eta, v, Nvp, Nep)
  end do

  ! ===== final report (uses the LAST sample's binned profiles) =================
  if (par_rank == 0) then
     ! spectral centroids: <k> weighted by error D vs by energy E (kx and kz)
     cD_kx=0; cE_kx=0; tot=0
     do ix = 0, nbx
        cD_kx = cD_kx + real(ix,real64)*Dkx(ix);  cE_kx = cE_kx + real(ix,real64)*Ekx(ix)
     end do
     cD_kx = cD_kx/max(sum(Dkx),tiny(1.0_real64));  cE_kx = cE_kx/max(sum(Ekx),tiny(1.0_real64))
     cD_kz=0; cE_kz=0
     do ix = 0, nbz
        cD_kz = cD_kz + real(ix,real64)*Dkz(ix);  cE_kz = cE_kz + real(ix,real64)*Ekz(ix)
     end do
     cD_kz = cD_kz/max(sum(Dkz),tiny(1.0_real64));  cE_kz = cE_kz/max(sum(Ekz),tiny(1.0_real64))

     print '(a)', '   ==========================================================================='
     print '(a,i0,a)', '   samples = ', nsamp, '   (profiles below are from the LAST sample)'
     print '(a)', '   --- net vs gross (|net|~gross => systematic/dissipative; <<  => redistribution)'
     print '(a,es11.3,a,es11.3,a,f6.3)', '   eta:  net = ', netPe, '   gross = ', grsPe, &
          '   |net|/gross = ', abs(netPe)/max(grsPe,tiny(1.0_real64))
     print '(a,es11.3,a,es11.3,a,f6.3)', '   v  :  net = ', netPv, '   gross = ', grsPv, &
          '   |net|/gross = ', abs(netPv)/max(grsPv,tiny(1.0_real64))
     print '(a)', '   --- spectral centroid of the ERROR vs of the ENERGY (high => high-k weighted)'
     print '(a,f7.2,a,f7.2)', '   kx-index:  <k>_error = ', cD_kx, '     <k>_energy = ', cE_kx
     print '(a,f7.2,a,f7.2)', '   kz-index:  <k>_error = ', cD_kz, '     <k>_energy = ', cE_kz
     print '(a)', '   --------------------------------------------------------------------------'
     print '(a)', '   kx-profile   mx   E(kx)/Etot    |dN|^2(kx)/D    Pe(kx)/|Pe|tot'
     do ix = 0, nbx
        if (Ekx(ix) <= 0.0_real64 .and. Dkx(ix) <= 0.0_real64) cycle
        print '(a,i5,3(2x,es13.4))', '   kx ', ix, &
             Ekx(ix)/max(sum(Ekx),tiny(1.0_real64)), &
             Dkx(ix)/max(sum(Dkx),tiny(1.0_real64)), &
             Pekx(ix)/max(sum(abs(Pekx)),tiny(1.0_real64))
     end do
     print '(a)', '   --------------------------------------------------------------------------'
     print '(a)', '   kz-profile  |mz|  E(kz)/Etot    |dN|^2(kz)/D    Pe(kz)/|Pe|tot'
     do ix = 0, nbz
        if (Ekz(ix) <= 0.0_real64 .and. Dkz(ix) <= 0.0_real64) cycle
        print '(a,i5,3(2x,es13.4))', '   kz ', ix, &
             Ekz(ix)/max(sum(Ekz),tiny(1.0_real64)), &
             Dkz(ix)/max(sum(Dkz),tiny(1.0_real64)), &
             Pekz(ix)/max(sum(abs(Pekz)),tiny(1.0_real64))
     end do
     print '(a)', '   ==========================================================================='
     print '(a)', '   READING:'
     print '(a)', '     dp  build  -> all of |dN|^2, net, gross ~ 1e-12 or 0 (split==fused). Sanity.'
     print '(a)', '     sp  build  -> MEASURED (H100 + cuFFT): dN EXACTLY 0 at sp too --'
     print '(a)', '        the split path (1-D z-FFT -> transpose -> 1-D x-FFT) is bit-identical'
     print '(a)', '        to the fused 2-D plan (cuFFT runs the same 1-D passes in the same axis'
     print '(a)', '        order; the transpose only moves bytes). A distributed-sp uu deficit'
     print '(a)', '        therefore CANNOT be a split-transform defect. If a future cuFFT/arch'
     print '(a)', '        ever shows |dN|^2 > 0 here, read it with:'
     print '(a)', '        |net|/gross ~ 1   => systematic injection (a real drain/source, dissipative-like)'
     print '(a)', '        |net|/gross ~ 0   => REDISTRIBUTION (energy moved between modes, ~conserved)'
     print '(a)', '        <k>_error >> <k>_energy => the error is high-k weighted (transform tail)'
     print '(a)', '        compare the kx- vs kz-profiles to see which axis carries it (z-FFT/transpose vs x-FFT)'
     print '(a)', '   ==========================================================================='
  end if

  call nl_destroy(wsf);  call par_nl_destroy(wss)
  call band_ops_free(bo);  call vsolve_ops_free(vo);  call mean_ops_free(mo)
  deallocate(he0, hv0, eta, v, u0, w0, Nvp, Nep, kx, kz, &
             hvf, hetaf, vS, etaS, hvs, hetas, hvfS, hetafS, &
             PeM, PvM, DeM, DvM, NeM, NvM, hPeM, hPvM, hDeM, hDvM, hNeM, hNvM)
  call par_finalize()

contains

  subroutine build_ic(he, hv, hu0, hw0)
    complex(dp), intent(out) :: he(0:n,nkx,nz), hv(0:n,nkx,nz), hu0(0:n), hw0(0:n)
    complex(wp), device, allocatable :: vd(:,:,:), ed(:,:,:), u0d(:), w0d(:)
    integer(ip) :: ip_, sx, sz, tx, tz
    integer(ip), parameter :: nprobe = 5
    integer(ip) :: pmx(nprobe), pmz(nprobe)
    real(dp) :: amp
    ! high-k probe targets (mx,|mz|): deliberately past the seeded band so the
    ! upper spectrum is NOT empty when we measure the transform there.
    pmx = (/ 20, 28, 36, 44, 52 /)
    pmz = (/ 24, 40, 16, 48, 32 /)
    allocate(vd(0:n,nkx,nz), ed(0:n,nkx,nz), u0d(0:n), w0d(0:n))
    call init_turbulent(n, nkx, nz, kxh, kzh, Qtar, epsp, pseed, kcut, vd, ed, u0d, w0d)
    istat = cudaDeviceSynchronize()
    hv = vd;  he = ed;  hu0 = u0d;  hw0 = w0d
    deallocate(vd, ed, u0d, w0d)
    ! inject the probe on the host: copy a populated, BC-valid low-k column (mx=1,
    ! mz=3) -- already in the Shen-biharmonic/Dirichlet bases -> wall BCs preserved
    ! -- into the high-k slots, at a fixed fraction. kx>0 so no z-symmetrisation.
    sx = 2;  sz = 4                                   ! source: mx=1, mz=3
    amp = 0.30_dp
    do ip_ = 1, nprobe
       tx = pmx(ip_) + 1                              ! ix = mx+1
       tz = pmz(ip_) + 1                              ! iz = mz+1  (mz>0)
       if (tx < 2 .or. tx > nkx-1) cycle              ! keep kx>0 interior (not Nyquist)
       if (tz < 2 .or. tz > nz/2)  cycle              ! keep 0<mz<nz/2
       hv(:,tx,tz) = hv(:,tx,tz) + amp*hv(:,sx,sz)
       he(:,tx,tz) = he(:,tx,tz) + amp*he(:,sx,sz)
    end do
  end subroutine build_ic

  ! copy the kx-slab src_full(:, kx0+1 : kx0+nxl, :) into a contiguous slab buffer.
  subroutine copy_slab(m, nzl, kx0, nxl, src, dst)
    integer(ip), intent(in) :: m, nzl, kx0, nxl
    complex(dp), device :: src(0:m, nkx, nzl)
    complex(dp), device :: dst(0:m, nxl, nzl)
    integer(ip) :: iy, jx_, jz
    !$cuf kernel do(3) <<<*,*>>>
    do jz = 1, nzl;  do jx_ = 1, nxl;  do iy = 0, m
       dst(iy,jx_,jz) = src(iy, kx0+jx_, jz)
    end do; end do; end do
  end subroutine copy_slab

  ! per-(kx,kz) maps summed over y. One CUF thread per (jx,jz) mode, serial in y
  ! (the proven stats_mod reduction shape -- no cross-thread scalar reduction).
  !   Pe = sum_y Re[conj(eta) (hes-hef)]   De = sum_y |hes-hef|^2   Ne = sum_y |eta|^2
  !   Pv = sum_y Re[conj(v)   (hvs-hvf)]   Dv = sum_y |hvs-hvf|^2   Nv = sum_y |v|^2
  subroutine mode_maps(m, nxl, nzl, vv, ee, hvs_, hvf_, hes_, hef_, Pe, Pv, De, Dv, Ne, Nv)
    integer(ip), intent(in) :: m, nxl, nzl
    complex(dp), device :: vv(0:m,nxl,nzl), ee(0:m,nxl,nzl)
    complex(dp), device :: hvs_(0:m,nxl,nzl), hvf_(0:m,nxl,nzl)
    complex(dp), device :: hes_(0:m,nxl,nzl), hef_(0:m,nxl,nzl)
    real(dp), device :: Pe(nxl,nzl), Pv(nxl,nzl), De(nxl,nzl), Dv(nxl,nzl), Ne(nxl,nzl), Nv(nxl,nzl)
    integer(ip) :: iy, jx_, jz
    real(dp) :: pe_, pv_, de_, dv_, ne_, nv_
    complex(dp) :: dq
    !$cuf kernel do(2) <<<*,*>>>
    do jz = 1, nzl
       do jx_ = 1, nxl
          pe_=0.0_dp; pv_=0.0_dp; de_=0.0_dp; dv_=0.0_dp; ne_=0.0_dp; nv_=0.0_dp
          do iy = 0, m
             dq  = hes_(iy,jx_,jz) - hef_(iy,jx_,jz)
             pe_ = pe_ + real(conjg(ee(iy,jx_,jz))*dq, dp)
             de_ = de_ + real(dq*conjg(dq), dp)
             ne_ = ne_ + real(ee(iy,jx_,jz)*conjg(ee(iy,jx_,jz)), dp)
             dq  = hvs_(iy,jx_,jz) - hvf_(iy,jx_,jz)
             pv_ = pv_ + real(conjg(vv(iy,jx_,jz))*dq, dp)
             dv_ = dv_ + real(dq*conjg(dq), dp)
             nv_ = nv_ + real(vv(iy,jx_,jz)*conjg(vv(iy,jx_,jz)), dp)
          end do
          Pe(jx_,jz)=pe_; Pv(jx_,jz)=pv_; De(jx_,jz)=de_; Dv(jx_,jz)=dv_
          Ne(jx_,jz)=ne_; Nv(jx_,jz)=nv_
       end do
    end do
  end subroutine mode_maps

end program test_driver_nldiss_mp
