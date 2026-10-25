!===============================================================================
! test_spec_helpers.F90  --  validate the batched GPU spectral helpers against
! their cheb_shen CPU equivalents (so the GPU projection/reconstruction match
! exactly what helm_band expects), plus the wall-derivative formula directly.
!
! Build:
!   nvfortran -c types.F90
!   nvfortran -c cheb_shen.F90
!   nvfortran -cuda -c spec_helpers_mod.F90
!   nvfortran -cuda -c test_spec_helpers.F90
!   nvfortran -cuda -o test_spec_helpers test_spec_helpers.o spec_helpers_mod.o \
!             cheb_shen.o types.o
!   ./test_spec_helpers
!===============================================================================
program test_spec_helpers
!  [CLAUSON layout] Canonical build: `bash run_all_tests.sh` from the repo
!  root (sources in ../src, output to ../build). The per-file nvfortran
!  commands in the header above list this test's MINIMAL module deps; under
!  the new layout they are: nvfortran ... -module build -c src/<mod>.F90 ...
  use kinds,            only: dp, ip
  use cheb_shen,        only: helmholtz_rhs, shen_to_cheb
  use spec_helpers_mod, only: project_shen, shen_to_cheb_c, wall_derivs_c
  use cudafor
  implicit none

  integer(ip) :: nv(3) = [12, 17, 32]
  integer(ip) :: xv(3) = [ 8, 12, 16]
  integer(ip) :: zv(3) = [ 6,  8, 12]
  integer(ip) :: c
  logical :: all_pass

  all_pass = .true.
  do c = 1, 3
     call run_grid(nv(c), xv(c), zv(c), all_pass)
  end do
  if (all_pass) then
     print *, 'ALL TESTS PASSED'
  else
     print *, 'SOME TESTS FAILED'
  end if

contains

  subroutine run_grid(n, nx, nz, all_pass)
    integer(ip), intent(in)    :: n, nx, nz
    logical,     intent(inout) :: all_pass
    ! host
    complex(dp), allocatable :: fc(:,:,:), uu(:,:,:), aa(:,:,:)
    complex(dp), allocatable :: ftil_ref(:,:,:), ach_ref(:,:,:)
    complex(dp), allocatable :: dlo_ref(:,:), dhi_ref(:,:)
    complex(dp), allocatable :: ftil_g(:,:,:), ach_g(:,:,:), dlo_g(:,:), dhi_g(:,:)
    real(dp) :: fre(0:n), fim(0:n), tre(0:n-2), tim(0:n-2)
    real(dp) :: ure(0:n-2), uim(0:n-2), are(0:n), aim(0:n)
    real(dp) :: slo_re, slo_im, shi_re, shi_im, rr, ri, jj
    integer(ip) :: ix, iz, k, j, s, istat
    ! device
    complex(dp), device, allocatable :: fc_d(:,:,:), uu_d(:,:,:), aa_d(:,:,:)
    complex(dp), device, allocatable :: ftil_d(:,:,:), ach_d(:,:,:), dlo_d(:,:), dhi_d(:,:)

    allocate(fc(0:n,nx,nz), uu(0:n-2,nx,nz), aa(0:n,nx,nz))
    allocate(ftil_ref(0:n-2,nx,nz), ach_ref(0:n,nx,nz), dlo_ref(nx,nz), dhi_ref(nx,nz))
    allocate(ftil_g(0:n-2,nx,nz), ach_g(0:n,nx,nz), dlo_g(nx,nz), dhi_g(nx,nz))

    block   ! [repro] fixed seed, applied once per run -> the suite is a bit-identity oracle
            ! across the precision phases (replaces a bare call random_seed())
      integer              :: rs_n_, rs_j_
      integer, allocatable :: rs_seed_(:)
      logical, save        :: rs_done_ = .false.
      if (.not. rs_done_) then
         call random_seed(size = rs_n_)
         allocate(rs_seed_(rs_n_))
         rs_seed_ = [ (1789 + 101*rs_j_, rs_j_ = 0, rs_n_ - 1) ]
         call random_seed(put = rs_seed_)
         rs_done_ = .true.
      end if
    end block
    call fill(fc);  call fill_s(uu);  call fill(aa)

    ! ---- CPU references ----
    do iz = 1, nz
       do ix = 1, nx
          ! project: helmholtz_rhs on re/im
          do k = 0, n
             fre(k) = real(fc(k,ix,iz)); fim(k) = aimag(fc(k,ix,iz))
          end do
          call helmholtz_rhs(n, fre, tre)
          call helmholtz_rhs(n, fim, tim)
          do k = 0, n-2
             ftil_ref(k,ix,iz) = cmplx(tre(k), tim(k), dp)
          end do
          ! shen_to_cheb on re/im
          do k = 0, n-2
             ure(k) = real(uu(k,ix,iz)); uim(k) = aimag(uu(k,ix,iz))
          end do
          call shen_to_cheb(n, ure, are)
          call shen_to_cheb(n, uim, aim)
          do k = 0, n
             ach_ref(k,ix,iz) = cmplx(are(k), aim(k), dp)
          end do
          ! wall derivs of aa, directly
          slo_re=0; slo_im=0; shi_re=0; shi_im=0
          do j = 1, n
             jj = real(j*j, dp)
             shi_re = shi_re + real(aa(j,ix,iz))*jj;  shi_im = shi_im + aimag(aa(j,ix,iz))*jj
             if (mod(j,2)==1) then; s = 1; else; s = -1; end if
             slo_re = slo_re + real(s,dp)*real(aa(j,ix,iz))*jj
             slo_im = slo_im + real(s,dp)*aimag(aa(j,ix,iz))*jj
          end do
          dhi_ref(ix,iz) = cmplx(shi_re, shi_im, dp)
          dlo_ref(ix,iz) = cmplx(slo_re, slo_im, dp)
       end do
    end do

    ! ---- GPU ----
    allocate(fc_d(0:n,nx,nz), uu_d(0:n-2,nx,nz), aa_d(0:n,nx,nz))
    allocate(ftil_d(0:n-2,nx,nz), ach_d(0:n,nx,nz), dlo_d(nx,nz), dhi_d(nx,nz))
    fc_d = fc; uu_d = uu; aa_d = aa
    call project_shen  (n, nx, nz, fc_d, ftil_d)
    call shen_to_cheb_c(n, nx, nz, uu_d, ach_d)
    call wall_derivs_c (n, nx, nz, aa_d, dlo_d, dhi_d)
    istat = cudaDeviceSynchronize()
    ftil_g = ftil_d; ach_g = ach_d; dlo_g = dlo_d; dhi_g = dhi_d

    call cmp3('project_shen   vs helmholtz_rhs', n-2, nx, nz, ftil_g, ftil_ref, all_pass)
    call cmp3('shen_to_cheb_c vs shen_to_cheb ', n,   nx, nz, ach_g,  ach_ref,  all_pass)
    call cmp2('wall_derivs_c  Du(-1)          ', nx, nz, dlo_g, dlo_ref, all_pass)
    call cmp2('wall_derivs_c  Du(+1)          ', nx, nz, dhi_g, dhi_ref, all_pass)

    deallocate(fc,uu,aa,ftil_ref,ach_ref,dlo_ref,dhi_ref,ftil_g,ach_g,dlo_g,dhi_g)
    deallocate(fc_d,uu_d,aa_d,ftil_d,ach_d,dlo_d,dhi_d)
  end subroutine run_grid

  subroutine fill(x)
    complex(dp), intent(out) :: x(:,:,:)
    integer :: i,j,k; real(dp) :: a,b
    do k=1,size(x,3); do j=1,size(x,2); do i=1,size(x,1)
       call random_number(a); call random_number(b); x(i,j,k)=cmplx(a-0.5_dp,b-0.5_dp,dp)
    end do; end do; end do
  end subroutine
  subroutine fill_s(x)
    complex(dp), intent(out) :: x(:,:,:)
    call fill(x)
  end subroutine

  subroutine cmp3(name, mtop, nx, nz, g, r, all_pass)
    character(*), intent(in) :: name
    integer(ip),  intent(in) :: mtop, nx, nz
    complex(dp),  intent(in) :: g(0:mtop,nx,nz), r(0:mtop,nx,nz)
    logical, intent(inout)   :: all_pass
    real(dp) :: err, den, d; logical :: bad; integer :: ix,iz,k
    err=0; den=0; bad=.false.
    do iz=1,nz; do ix=1,nx; do k=0,mtop
       d=abs(g(k,ix,iz)-r(k,ix,iz)); if(.not.(d==d)) bad=.true.
       err=max(err,d); den=max(den,abs(r(k,ix,iz)))
    end do; end do; end do
    call rep(name, merge(1.0e30_dp, err/merge(den,1.0_dp,den/=0.0_dp), bad), all_pass)
  end subroutine
  subroutine cmp2(name, nx, nz, g, r, all_pass)
    character(*), intent(in) :: name
    integer(ip),  intent(in) :: nx, nz
    complex(dp),  intent(in) :: g(nx,nz), r(nx,nz)
    logical, intent(inout)   :: all_pass
    real(dp) :: err, den, d; logical :: bad; integer :: ix,iz
    err=0; den=0; bad=.false.
    do iz=1,nz; do ix=1,nx
       d=abs(g(ix,iz)-r(ix,iz)); if(.not.(d==d)) bad=.true.
       err=max(err,d); den=max(den,abs(r(ix,iz)))
    end do; end do
    call rep(name, merge(1.0e30_dp, err/merge(den,1.0_dp,den/=0.0_dp), bad), all_pass)
  end subroutine
  subroutine rep(name, e, all_pass)
    character(*), intent(in) :: name
    real(dp), intent(in) :: e
    logical, intent(inout) :: all_pass
    character(6) :: tag
    if (e <= 1.0e-12_dp) then; tag='[PASS]'; else; tag='[FAIL]'; all_pass=.false.; end if
    write(*,'(a,2x,a,2x,a,es12.4)') tag, name, 'rel err=', e
  end subroutine

end program test_spec_helpers
