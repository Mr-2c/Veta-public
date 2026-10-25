!===============================================================================
! test_par_transpose.F90  --  MULTI-GPU validation of par_transpose_mod (the
! real NCCL A<->B transpose).
!
! Serial-vs-parallel bit-identity: a transpose
! only MOVES data, so the distributed result must equal the serial layout
! EXACTLY (tol = 0), with no float reassociation.  We avoid any rank-0 broadcast
! by defining the global field as a DETERMINISTIC function of the GLOBAL index
!   G(iy,kx,iz,f) = cmplx(L, -L)  with  L = (iy-1) + ny*((kx-1)+nkx*((iz-1)+nz*(f-1)))
! L is an integer < 2^24, hence EXACT in float32 and distinct per element, so any
! mis-mapped element is caught bit-exactly under tp = sp or dp.
!
! Each rank:
!   * fills ITS layout-A z-slab from G (full kx, its z range)         [host->dev]
!   * par_transpose_A_to_B  -> its layout-B kx-slab                   [NCCL]
!   * compares to an INDEPENDENT reference layout-B slab built from G  [bit-exact]
!   * par_transpose_B_to_A back -> layout A, compares to the original  [round trip]
! NaN-hardened (all(X==X)), nonzero-guarded (max|A|>0).  par_reduce_and makes any
! single rank's failure fail the whole job.  Runs on whatever world size the
! launcher provides (grid extents nkx=17, nz=16 keep blocks non-empty to 16 ranks;
! the kernels handle empty blocks too, validated in the model).
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_par_transpose   Run: bash run_all_par_tests.sh run -k test_par_transpose in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_par_transpose
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,             only: tp, ip
  use cudafor
  use decomp_mod,        only: decomp_t, decomp_setup
  use par_transpose_mod, only: par_transpose_A_to_B, par_transpose_B_to_A
  use par_mod
  implicit none

  integer(ip), parameter :: nx = 32, ny = 24, nz = 16       ! nkx = nx/2+1 = 17
  integer(ip), parameter :: nkx_g = nx/2 + 1
  integer(ip), parameter :: nfs(2) = [ 1_ip, 3_ip ]

  logical           :: all_ok = .true., ok
  integer           :: jf
  integer(ip)       :: nf
  character(len=72) :: lbl

  call par_init()
  if (par_rank == 0) print '(a,i0,a)', '[par-transpose] world size = ', par_nproc, ' rank(s)'

  do jf = 1, size(nfs)
     nf = nfs(jf)
     ok = run_one(nf)
     call par_reduce_and(ok)                       ! global AND across ranks
     if (.not. ok) all_ok = .false.
     if (par_rank == 0) then
        write(lbl,'(a,i0)') 'serial-vs-parallel A<->B bit-identity, nfield=', nf
        call report(trim(lbl), ok)
     end if
  end do

  call par_reduce_and(all_ok)
  if (par_rank == 0) then
     if (all_ok) then
        print '(a)', 'ALL TESTS PASSED'
     else
        print '(a)', 'SOME TESTS FAILED'
     end if
  end if

  call par_finalize()
  if (.not. all_ok) error stop 1

contains

  subroutine report(name, okk)
    character(*), intent(in) :: name
    logical,      intent(in) :: okk
    character(len=4) :: tag
    tag = merge('PASS', 'FAIL', okk)
    print '(a,a,a)', '[', tag, '] '//name
  end subroutine report

  ! deterministic global field value at GLOBAL indices (iy,kxg,izg,f) ----------
  complex(tp) function gval(iy, kxg, izg, f, nf) result(v)
    integer(ip), intent(in) :: iy, kxg, izg, f, nf
    integer(ip) :: L
    real(tp)    :: x
    L = (iy-1) + ny*((kxg-1) + nkx_g*((izg-1) + nz*(f-1)))
    x = real(L, tp)
    v = cmplx(x, -x, tp)
  end function gval

  ! one (nfield) case: distributed transpose vs independent reference + round trip
  logical function run_one(nf) result(good)
    integer(ip), intent(in) :: nf
    type(decomp_t) :: d
    integer(ip) :: az0, anz, kx0, bnkx
    integer(ip) :: iy, ikx, izl, izg, kxl, f
    integer     :: istat
    real(real64) :: maxd, maxa
    complex(tp), allocatable :: A_h(:,:,:,:), B_h(:,:,:,:), R_h(:,:,:,:), B_ref(:,:,:,:)
    complex(tp), allocatable, device :: A_d(:,:,:,:), B_d(:,:,:,:), R_d(:,:,:,:)

    good = .true.
    call decomp_setup(d, 0_ip, int(par_nproc, ip), int(par_rank, ip), nx, ny, nz)
    az0 = d%a_z0;   anz  = d%a_nz_loc           ! this rank's z-slab (layout A)
    kx0 = d%b_kx0;  bnkx = d%b_nkx_loc          ! this rank's kx-slab (layout B)

    allocate(A_h(ny,nkx_g,anz,nf), R_h(ny,nkx_g,anz,nf))
    allocate(B_h(ny,bnkx,nz,nf),   B_ref(ny,bnkx,nz,nf))
    allocate(A_d(ny,nkx_g,anz,nf), R_d(ny,nkx_g,anz,nf), B_d(ny,bnkx,nz,nf))

    ! this rank's layout-A slab from G (full kx, its z range)
    do f = 1, nf
       do izl = 1, anz
          do ikx = 1, nkx_g
             do iy = 1, ny
                A_h(iy,ikx,izl,f) = gval(iy, ikx, az0+izl, f, nf)
             end do
          end do
       end do
    end do
    ! independent reference layout-B slab from G (its kx range, full z)
    do f = 1, nf
       do izg = 1, nz
          do kxl = 1, bnkx
             do iy = 1, ny
                B_ref(iy,kxl,izg,f) = gval(iy, kx0+kxl, izg, f, nf)
             end do
          end do
       end do
    end do
    maxa = real(maxval(abs(A_h)), real64)
    if (.not. (maxa > 0.0_real64)) good = .false.    ! nonzero guard

    A_d = A_h
    call par_transpose_A_to_B(d, ny, nkx_g, nz, anz, bnkx, nf, A_d, B_d)
    call par_transpose_B_to_A(d, ny, nkx_g, nz, anz, bnkx, nf, B_d, R_d)
    istat = cudaDeviceSynchronize()
    B_h = B_d;  R_h = R_d

    if (.not. all(B_h == B_h)) good = .false.        ! NaN guard
    if (.not. all(R_h == R_h)) good = .false.
    maxd = real(maxval(abs(B_h - B_ref)), real64)
    if (maxd /= 0.0_real64) good = .false.           ! A->B vs independent ref (bit-exact)
    maxd = real(maxval(abs(R_h - A_h)), real64)
    if (maxd /= 0.0_real64) good = .false.           ! A->B->A round trip (bit-exact)

    deallocate(A_h, B_h, R_h, B_ref, A_d, B_d, R_d)
  end function run_one

end program test_par_transpose
