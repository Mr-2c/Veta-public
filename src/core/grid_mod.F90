!===============================================================================
! grid_mod.F90  --  Static spectral geometry: Fourier wavenumbers, k^2, the
!                   wall-normal GLC nodes, 3/2 dealias sizes, and a Nyquist mask.
!
! INDEX / SIZE CONVENTIONS
!   x: real-to-complex.  nkx = nx/2 + 1 modes, index i = 0..nx/2.
!        kx(i) = 2*pi*i / Lx.
!   z: full complex.     nkz = nz modes in cuFFT order: index j = 0..nz-1 maps
!        to frequency  j        for j <= nz/2,
!                       j - nz   for j >  nz/2     (negative frequencies),
!        so kz(j) = 2*pi/Lz * (that frequency).
!   y: ny GLC points -> Chebyshev degree n = ny - 1; yglc(j) = cos(pi j/(ny-1)),
!        which runs +1 (j=0) .. -1 (j=ny-1). Half-channel height = 1.
!   Dealias (3/2 rule): nxd = 3*nx/2, nzd = 3*nz/2 (padded physical sizes used
!        by the transform layer). nx, nz must be even.
!
! PRECISION: kx, kz, mask are stored in wp (they multiply wp fields); k2 is dp
! (it feeds the FP64 operator build) and is computed from dp wavenumbers so it
! is accurate regardless of wp. Built on the host, then assigned whole-array to
! the (possibly device) components -- one pattern that compiles for both the CPU
! test build and the -cuda GPU build.
!
! NOTE: the Shen operator matrices in grid_t (B_dir, ...) are NOT filled here;
! the solvers currently self-assemble (and are validated). Precomputing those
! dt-independent matrices into grid_t is a later optimization step.
!===============================================================================

module grid_mod
  use kinds, only: wp, dp, ip
  use types, only: grid_t, params_t
  use cheb_shen, only: glc_nodes
  implicit none
  private
  public :: build_grid, destroy_grid

  real(dp), parameter :: TWOPI = 2.0_dp*acos(-1.0_dp)

contains

  subroutine build_grid(grid, params, nx, ny, nz)
    type(grid_t),   intent(inout) :: grid
    type(params_t), intent(in)    :: params
    integer(ip),    intent(in)    :: nx, ny, nz
    real(dp), allocatable :: tkx(:), tkz(:), tk2(:,:), tmask(:,:), tyglc(:)
    real(dp) :: lx, lz, freq
    integer(ip) :: i, j

    if (mod(nx,2) /= 0 .or. mod(nz,2) /= 0) error stop 'build_grid: nx, nz must be even'

    grid%nx = nx;  grid%ny = ny;  grid%nz = nz
    grid%nkx = nx/2 + 1
    grid%nkz = nz
    grid%nxd = 3*nx/2
    grid%nzd = 3*nz/2
    lx = params%lx;  lz = params%lz

    allocate(grid%kx(0:grid%nkx-1), grid%kz(0:grid%nkz-1))
    allocate(grid%k2(0:grid%nkx-1, 0:grid%nkz-1))
    allocate(grid%mask(0:grid%nkx-1, 0:grid%nkz-1))
    allocate(grid%yglc(0:ny-1))

    allocate(tkx(0:grid%nkx-1), tkz(0:grid%nkz-1))
    allocate(tk2(0:grid%nkx-1, 0:grid%nkz-1))
    allocate(tmask(0:grid%nkx-1, 0:grid%nkz-1))
    allocate(tyglc(0:ny-1))

    ! x wavenumbers (positive only, real-to-complex)
    do i = 0, grid%nkx-1
       tkx(i) = TWOPI*real(i,dp)/lx
    end do

    ! z wavenumbers (cuFFT order: positive then negative)
    do j = 0, grid%nkz-1
       if (j <= nz/2) then
          freq = real(j,dp)
       else
          freq = real(j-nz,dp)
       end if
       tkz(j) = TWOPI*freq/lz
    end do

    ! k^2 from dp wavenumbers
    do j = 0, grid%nkz-1
       do i = 0, grid%nkx-1
          tk2(i,j) = tkx(i)*tkx(i) + tkz(j)*tkz(j)
       end do
    end do

    ! dealias / Nyquist mask: 1 everywhere, 0 on the Nyquist lines
    tmask = 1.0_dp
    tmask(nx/2, :) = 0.0_dp
    tmask(:, nz/2) = 0.0_dp

    ! wall-normal GLC nodes (degree n = ny-1)
    call glc_nodes(ny-1, tyglc)

    ! host -> (possibly device) components; cast dp->wp where needed
    grid%kx   = real(tkx,  wp)
    grid%kz   = real(tkz,  wp)
    grid%mask = real(tmask, wp)
    grid%k2   = tk2
    grid%yglc  = tyglc

    deallocate(tkx, tkz, tk2, tmask, tyglc)
  end subroutine build_grid

  subroutine destroy_grid(grid)
    type(grid_t), intent(inout) :: grid
    if (allocated(grid%kx))   deallocate(grid%kx)
    if (allocated(grid%kz))   deallocate(grid%kz)
    if (allocated(grid%k2))   deallocate(grid%k2)
    if (allocated(grid%mask)) deallocate(grid%mask)
    if (allocated(grid%yglc))  deallocate(grid%yglc)
  end subroutine destroy_grid

end module grid_mod
