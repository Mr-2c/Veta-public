!===============================================================================
! diag_mod.F90  --  startup diagnostics (no trajectory effect -- both routines
! only PRINT). Two concerns:
!
!   (1) report_device_memory : per-category device-memory footprint, broken down
!       by the three-precision TIERS (tp/wp/dp) of src/core/types.F90, with a
!       real64 / fp32-transform / fp32-transform+fields projection so the
!       memory trade of each operating point is visible at a glance. Catches
!       scratch-count regressions at scale and quantifies the fp32-transform
!       (~tp) and fp32-transform+fields (~tp+wp) savings.
!
!   (2) fft_size_guard : warn if any Fourier transform length is not 7-smooth
!       (largest prime factor <= 7). cuFFT is efficient for radices {2,3,5,7};
!       a large prime factor forces the slow Bluestein path. Advisory only.
!
! IMPORTANT (tiers vs the current code): the footprint is computed at the TARGET
! tier assignment, NOT the kind each array currently carries. Under the
! fp32-transform tier (tp=sp) the transform line is forward-looking until
! nl_workspace is retyped from wp to tp; this is flagged in the output.
! SCOPE OF THE TOTAL: the inventory covers
! the MAJOR persistent tiers (fields/history, nl_workspace, stepper_ws +
! module deriv/band/FFT/x-leading scratch, factorization, small solver setup).
! It does NOT itemize the mean_gpu 1-D state, the FD band/cache tiers, or the
! cuFFT arena (the driver banner prints the arena separately), so the printed
! total UNDERSTATES nvidia-smi by those small tiers -- close and honest, not
! byte-exact.
! Under FD-y the report OVERSTATES the run's scratch by: the cheb
! deriv/FFT/hb tiers (never allocated there) AND the 3 Shen
! stepper buffers still counted in the dp solve tier (N_STEP_DP_CSHEN --
! stepper_ws_alloc(fd=.true.) allocates them as 1-element stubs). The
! reporter has no basis input; the FD surplus is a known, listed set.
!
! The byte arithmetic lives in the PURE subroutine mem_category_bytes (unit-
! tested in tests/serial/test_diag.F90); report_device_memory is just kind-probe +
! formatting around it. largest_prime_factor is likewise pure and tested.
!
! CPU-only module (host prints + integer arithmetic); no device code. Depends
! only on `kinds`. Build: nvfortran -c diag_mod.F90  (no -cuda needed).
!===============================================================================
module diag_mod
  use kinds, only: tp, wp, dp, ip
  use iso_fortran_env, only: int64, output_unit
  implicit none
  private
  public :: report_device_memory, fft_size_guard, mem_category_bytes, largest_prime_factor

  ! number of device arrays of each shape, per category (from the verified
  ! inventory: main.F90 fields/history/factorization + stepper_ws + nl_workspace).
  ! "full" = (0:n , nkx, nz) ; "shen" = (0:n-2, nkx, nz) ; "phys" = (0:n , nxd, nzd) real ;
  ! "fine" = (0:n , nkxd, nzd) complex ; nc = n+1 ; ns = n-1.
  integer(ip), parameter :: N_TRANSFORM_CFULL = 12   ! nl_workspace uh,wh,ox,oy,oz,dv,du,dw,gg,Hx,Hy,Hz
  integer(ip), parameter :: N_TRANSFORM_RPHYS = 9    ! nl_workspace pu,pv,pw,pox,poy,poz,phx,phy,phz
  integer(ip), parameter :: N_TRANSFORM_CFINE = 1    ! nl_workspace Wf
  integer(ip), parameter :: N_FIELD_WP_CFULL  = 6    ! eta,v,Nv_prev,Neta_prev,v_good,eta_good
  integer(ip), parameter :: N_FIELD_DP_CMEAN  = 6    ! u0,w0,Nu0_prev,Nw0_prev,u0_good,w0_good  (0:n)
  integer(ip), parameter :: N_STEP_WP_CFULL   = 6    ! Nv_cur,Neta_cur,rhs,scr,Lv,rhsv
  integer(ip), parameter :: N_STEP_WP_CMEAN   = 2    ! hx0,hz0  (0:n)
  integer(ip), parameter :: N_STEP_DP_CFULL   = 0    ! none (the vif path lends rhs/Lv)
  integer(ip), parameter :: N_STEP_DP_CSHEN   = 3    ! ftil,vgsh,vvsh                (0:n-2)
  integer(ip), parameter :: N_STEP_DP_RSHEN   = 3    ! dia,up1,up2            (0:n-2) real
  integer(ip), parameter :: N_STEP_DP_CPLANE  = 2    ! vdlo,vdhi              (nkx,nz)
  integer(ip), parameter :: N_FACT_DP_RSTAGE  = 2    ! vm_all,vp_all          (0:n,nkx,nz,3) real
  ! the module-persistent scratch tiers (grow-only, sized on first use at
  ! the shapes below; the FD-y backend allocates NONE of the cheb ones):
  integer(ip), parameter :: N_CD_WP_CFULL     = 2    ! cheb_ops cdwp_a/b      (nkx*nz, n+1) complex wp
  integer(ip), parameter :: N_CD_TP_CFULL     = 2    ! cheb_ops cdtp_a/b      (nkx*nz, n+1) complex tp
  integer(ip), parameter :: N_HB_DP_CSHEN     = 2    ! helm_band hb_fcl/ucl   (nkx*nz, 0:n-2) complex dp
  ! (shapes are the SERIAL extents this reporter serves; the parallel driver
  ! sizes the same scratch with its slab-local batch and has no reporter)
  ! plus, term-wise below: cheb_fft gc_buf (0:2n-1, nkx*nz) complex tp.
  ! NOT counted: the fft3d x-leading slot pairs -- they are
  ! allocated only via exec_x_r2c/c2r, whose only callers are in the PARALLEL
  ! transform path (the serial path is the fused exec_xz_*), so a serial run
  ! never allocates them. Still NOT itemized (small): mean_gpu 1-D state, FD
  ! band/cache tiers, the cuFFT arena (printed separately by the driver
  ! banner).

contains

  !-- per-category device bytes at the given real-kind byte sizes (PURE; tested). -
  ! rb_tp / rb_wp / rb_dp are the byte sizes of one real of kind tp / wp / dp
  ! (i.e. storage_size(...)/8). Complex arrays count as 2*rb. int64 throughout to
  ! avoid overflow at production sizes. b_misc = the small dp solver-setup arrays.
  pure subroutine mem_category_bytes(n, nkx, nz, nxd, nzd, rb_tp, rb_wp, rb_dp, &
                                     b_transform, b_fields, b_step_wp, b_step_dp, &
                                     b_factor, b_misc)
    integer(ip),    intent(in)  :: n, nkx, nz, nxd, nzd
    integer(ip),    intent(in)  :: rb_tp, rb_wp, rb_dp
    integer(int64), intent(out) :: b_transform, b_fields, b_step_wp, b_step_dp, b_factor, b_misc
    integer(int64) :: nc, ns, nkxd, c_nkx, c_nz, full, shen, phys, fine, plane, mean
    integer(int64) :: tpb, wpb, dpb, n2
    nc    = int(n,int64) + 1_int64
    ns    = int(n,int64) - 1_int64
    c_nkx = int(nkx,int64);  c_nz = int(nz,int64)
    nkxd  = int(nxd,int64)/2_int64 + 1_int64
    full  = nc*c_nkx*c_nz                          ! (0:n , nkx, nz)
    shen  = ns*c_nkx*c_nz                           ! (0:n-2, nkx, nz)
    phys  = nc*int(nxd,int64)*int(nzd,int64)        ! (0:n , nxd, nzd)  real
    fine  = nc*nkxd*int(nzd,int64)                  ! (0:n , nkxd, nzd) complex
    plane = c_nkx*c_nz                              ! (nkx, nz)
    mean  = nc                                      ! (0:n)
    n2    = 2_int64*(nc - 1_int64)                  ! cheb even-extension length 2n
    tpb = int(rb_tp,int64);  wpb = int(rb_wp,int64);  dpb = int(rb_dp,int64)

    ! TRANSFORMS (tp): nl_workspace = complex full*12 + real phys*9 + complex fine*1 + kx/kz
    ! + the tp module scratch: cheb_ops cdtp_a/b (2
    ! complex full) and cheb_fft gc_buf (complex, 2n batch). No xl terms:
    ! serial runs never allocate the x-leading slots (see the constants note).
    b_transform = tpb*( 2_int64*int(N_TRANSFORM_CFULL,int64)*full &
                      +         int(N_TRANSFORM_RPHYS,int64)*phys &
                      + 2_int64*int(N_TRANSFORM_CFINE,int64)*fine &
                      + 2_int64*int(N_CD_TP_CFULL,int64)*full &
                      + 2_int64*(n2*c_nkx*c_nz) &
                      + (c_nkx + c_nz) )

    ! FIELDS + HISTORY: 6 complex full @ wp ; 6 complex (0:n) @ dp
    b_fields = wpb*( 2_int64*int(N_FIELD_WP_CFULL,int64)*full ) &
             + dpb*( 2_int64*int(N_FIELD_DP_CMEAN,int64)*mean )

    ! STEPPER assembly (wp): 6 complex full + 2 complex (0:n)
    ! + cheb_ops cdwp_a/b (2 complex full wp, the apply_L deriv scratch)
    b_step_wp = wpb*( 2_int64*int(N_STEP_WP_CFULL,int64)*full &
                    + 2_int64*int(N_CD_WP_CFULL,int64)*full &
                    + 2_int64*int(N_STEP_WP_CMEAN,int64)*mean )

    ! STEPPER solve/band (dp): 3+2 complex shen + 3 real shen + 2 complex plane
    ! (the CFULL term is kept in the formula with count 0;
    ! + helm_band hb_fcl/ucl, 2 complex shen)
    b_step_dp = dpb*( 2_int64*int(N_STEP_DP_CFULL,int64)*full &
                    + 2_int64*(int(N_STEP_DP_CSHEN,int64) + int(N_HB_DP_CSHEN,int64))*shen &
                    +         int(N_STEP_DP_RSHEN,int64)*shen &
                    + 2_int64*int(N_STEP_DP_CPLANE,int64)*plane )

    ! FACTORIZATION (dp): vm_all+vp_all = real (0:n,nkx,nz,3) ; Cinv_all = real (2,2,nkx,nz,3)
    b_factor = dpb*( int(N_FACT_DP_RSTAGE,int64)*(full*3_int64) &
                   + (4_int64*plane*3_int64) )

    ! MISC solver setup (dp, small): kx+kz (nkx+nz) + sd,st,bd (3*ns) + inv_dy_d (nc)
    b_misc = dpb*( (c_nkx + c_nz) + 3_int64*ns + nc )
  end subroutine mem_category_bytes

  !-- largest prime factor of m (>=1). PURE; tested. lpf(1)=1, lpf(prime)=prime. --
  pure function largest_prime_factor(m) result(lpf)
    integer(ip), intent(in) :: m
    integer(ip) :: lpf, r, f
    r = m; lpf = 1_ip
    if (r <= 1_ip) return
    f = 2_ip
    do while (f*f <= r)
       do while (mod(r,f) == 0_ip)
          lpf = f; r = r/f
       end do
       f = f + 1_ip
    end do
    if (r > 1_ip) lpf = r        ! leftover prime
  end function largest_prime_factor

  !-- format bytes as GiB (real64) -----------------------------------------------
  pure function gib(b) result(g)
    integer(int64), intent(in) :: b
    real(dp) :: g
    g = real(b,dp)/ (1024.0_dp**3)
  end function gib

  !-- one projection column: print per-category + total for given kind bytes ------
  subroutine print_column(label, n, nkx, nz, nxd, nzd, rb_tp, rb_wp, rb_dp)
    character(*), intent(in) :: label
    integer(ip),  intent(in) :: n, nkx, nz, nxd, nzd, rb_tp, rb_wp, rb_dp
    integer(int64) :: bt, bf, bsw, bsd, bfa, bm, tot
    call mem_category_bytes(n, nkx, nz, nxd, nzd, rb_tp, rb_wp, rb_dp, bt, bf, bsw, bsd, bfa, bm)
    tot = bt + bf + bsw + bsd + bfa + bm
    write(output_unit,'(2x,a14,1x,6(f10.4,1x),f11.4)') label, &
         gib(bt), gib(bf), gib(bsw), gib(bsd), gib(bfa), gib(bm), gib(tot)
  end subroutine print_column

  !-- startup memory report ------------------------------------------------------
  ! Prints the ACTIVE build's footprint plus a real64 / fp32-transform /
  ! fp32-transform+fields projection. Excludes the optional stats/divergence diagnostic buffers and the
  ! tiny host arrays. n = Chebyshev degree; nkx = nx/2+1; nxd/nzd = dealiased.
  subroutine report_device_memory(n, nkx, nz, nxd, nzd)
    integer(ip), intent(in) :: n, nkx, nz, nxd, nzd
    integer(ip) :: rt, rw, rd
    real(dp)    :: tot_r64, tot_dt, tot_rb
    integer(int64) :: bt, bf, bsw, bsd, bfa, bm
    rt = storage_size(1.0_tp)/8;  rw = storage_size(1.0_wp)/8;  rd = storage_size(1.0_dp)/8

    write(output_unit,'(/,a)') '==== device memory footprint (per precision tier) ===================='
    write(output_unit,'(2x,3(a,i0))') 'grid: n=',n,'   nkx=',nkx,'   nz=',nz
    write(output_unit,'(2x,a,2(i0,a))') '      nxd=',nxd,'  nzd=',nzd,' (dealiased)'
    write(output_unit,'(2x,3(a,i0))') 'active kinds (real bytes): tp=',rt,'  wp=',rw,'  dp=',rd
    write(output_unit,'(2x,a)') 'categories: transform(tp) fields(wp+dp) step-asm(wp) step-solve(dp) factor(dp) misc(dp)'
    write(output_unit,'(/,2x,a14,1x,6(a10,1x),a11)') '(GiB)', &
         'transform','fields','step-asm','step-solv','factor','misc','TOTAL'
    call print_column('ACTIVE', n, nkx, nz, nxd, nzd, rt, rw, rd)
    write(output_unit,'(2x,a)') '---- projection across operating points (independent of this build) ----'
    call print_column('real64',   n, nkx, nz, nxd, nzd, 8, 8, 8)
    call print_column('TP=sp',    n, nkx, nz, nxd, nzd, 4, 8, 8)
    call print_column('TP+WP=sp', n, nkx, nz, nxd, nzd, 4, 4, 8)

    ! savings vs real64
    call mem_category_bytes(n, nkx, nz, nxd, nzd, 8,8,8, bt,bf,bsw,bsd,bfa,bm)
    tot_r64 = gib(bt+bf+bsw+bsd+bfa+bm)
    call mem_category_bytes(n, nkx, nz, nxd, nzd, 4,8,8, bt,bf,bsw,bsd,bfa,bm)
    tot_dt  = gib(bt+bf+bsw+bsd+bfa+bm)
    call mem_category_bytes(n, nkx, nz, nxd, nzd, 4,4,8, bt,bf,bsw,bsd,bfa,bm)
    tot_rb  = gib(bt+bf+bsw+bsd+bfa+bm)
    if (tot_r64 > 0.0_dp) then
       write(output_unit,'(2x,a,f6.2,a,f6.2,a)') 'savings vs real64:  TP=sp ', &
            100.0_dp*(1.0_dp - tot_dt/tot_r64), ' %    TP+WP=sp ', &
            100.0_dp*(1.0_dp - tot_rb/tot_r64), ' %'
    end if
    write(output_unit,'(a,/)') '======================================================================'
  end subroutine report_device_memory

  !-- FFT small-prime guard (advisory) -------------------------------------------
  ! Warns if any Fourier length (nx, nz, nxd, nzd) has a prime factor > 7.
  ! n2cheb = the FIFTH executed FFT length, the 2n Chebyshev even
  ! extension (an unlucky n, e.g. 482 -> 964 = 4*241, silently takes
  ! Bluestein on ~29% of the transform FLOPs otherwise).
  ! Pass 0 to skip (FD-y mode: the leg never runs).
  subroutine fft_size_guard(nx, nz, nxd, nzd, n2cheb)
    integer(ip), intent(in) :: nx, nz, nxd, nzd, n2cheb
    logical :: any_bad
    any_bad = .false.
    call check_one('nx ', nx,  any_bad)
    call check_one('nz ', nz,  any_bad)
    call check_one('nxd', nxd, any_bad)
    call check_one('nzd', nzd, any_bad)
    if (n2cheb > 0) call check_one('2n ', n2cheb, any_bad)
    if (any_bad) then
       write(output_unit,'(2x,a)') '[fft] cuFFT is fastest for lengths factoring into {2,3,5,7};'
       write(output_unit,'(2x,a)') '      a larger prime factor forces the slower Bluestein path.'
    end if
  contains
    subroutine check_one(name, m, bad)
      character(*), intent(in)    :: name
      integer(ip),  intent(in)    :: m
      logical,      intent(inout) :: bad
      integer(ip) :: lpf
      lpf = largest_prime_factor(m)
      if (lpf > 7_ip) then
         write(output_unit,'(2x,a,a,i0,a,i0,a)') '[fft] WARNING: ', name, m, &
              ' has large prime factor ', lpf, ' (not 7-smooth).'
         bad = .true.
      end if
    end subroutine check_one
  end subroutine fft_size_guard

end module diag_mod
