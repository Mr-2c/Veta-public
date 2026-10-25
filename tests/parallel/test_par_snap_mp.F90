!===============================================================================
! test_par_snap_mp.F90 -- validate par_snap_write (physical u,v,w snapshot):
! the z-slab single-shift writer + downcast, independent of the transform.
!
! Each rank fills its COARSE z-slab of a deterministic global field (unique per
! field), writes u,v,w via par_snap_write, and rank 0 reads the whole file back as
! a plain stream and checks every value at the canonical (ny,nx,nz) on-disk order:
!   (1) fp64 output  -> bit-exact vs real(field, dp)        [layout/offset/header]
!   (2) fp32 output  -> bit-exact vs real(field, sp)        [downcast path]
! Run at P=1,2,4 (z-slab tiling, incl. unequal slabs: nz=18 -> e.g. 5+5+4+4 at P=4).
!
! Build: TP=dp|sp bash scripts/build/build_par.sh test_par_snap_mp   Run: bash run_all_par_tests.sh run -k test_par_snap_mp in a GPU allocation (scripts/sbatch/run_all_par_tests.sbatch runs the whole suite)
!===============================================================================
program test_par_snap_mp
  use, intrinsic :: iso_fortran_env, only: real64
  use kinds,      only: tp, sp, dp, ip
  use cudafor
  use decomp_mod, only: decomp_t, decomp_setup
  use par_mod,    only: par_init, par_finalize, par_rank, par_nproc, par_reduce_and
  use par_io_mod, only: par_snap_write
  implicit none

  integer(ip), parameter :: n = 16, nx = 24, nz = 18      ! ny = 17
  integer(ip) :: ny, anz, z0, iy, ix, iz, f, iu, ios, csh
  type(decomp_t) :: d
  real(tp), device, allocatable :: u(:,:,:), v(:,:,:), w(:,:,:)
  real(tp), allocatable :: hu(:,:,:), hv(:,:,:), hw(:,:,:)
  real(dp), allocatable :: d64(:,:,:)
  real(sp), allocatable :: d32(:,:,:)
  real(dp), allocatable :: yh(:), cdisk(:)                ! v3 x,y,z records
  real(dp) :: Re, t, Lx, Lz
  integer(ip) :: step
  logical :: ok, allok
  character(len=*), parameter :: f64 = 'build_par/test_snap64.bin'
  character(len=*), parameter :: f32 = 'build_par/test_snap32.bin'

  call par_init()
  ny = n + 1
  call decomp_setup(d, 0_ip, int(par_nproc,ip), int(par_rank,ip), nx, ny, nz)
  anz = d%a_nz_loc;  z0 = d%a_z0
  Re = 2800.0_dp;  t = 0.5_dp;  step = 100;  Lx = 6.2831853_dp;  Lz = 3.1415927_dp
  csh = (nx + ny + nz)*8                                  ! coord block bytes

  allocate(u(ny,nx,anz), v(ny,nx,anz), w(ny,nx,anz), hu(ny,nx,anz), hv(ny,nx,anz), hw(ny,nx,anz))
  do iz = 1, anz; do ix = 1, nx; do iy = 1, ny
     hu(iy,ix,iz) = fval(iy, ix, z0+iz, 0_ip)
     hv(iy,ix,iz) = fval(iy, ix, z0+iz, 1_ip)
     hw(iy,ix,iz) = fval(iy, ix, z0+iz, 2_ip)
  end do; end do; end do
  u = hu;  v = hv;  w = hw
  allocate(yh(0:n))                                       ! deterministic "node" values
  do iy = 0, n
     yh(iy) = 1.0_dp - 2.0_dp*real(iy,dp)/real(n,dp)
  end do

  allok = .true.

  ! (1) fp64 output -> bit-exact (fields start after the x,y,z records)
  call par_snap_write(f64, d, n, nx, nz, Re, t, step, 8_ip, u, v, w, yh, Lx, Lz, ok)
  if (.not. ok) allok = .false.
  if (par_rank == 0) then
     allocate(d64(ny,nx,nz), cdisk(nx+ny+nz))
     open(newunit=iu, file=f64, access='stream', form='unformatted', status='old', action='read', iostat=ios)
     if (ios /= 0) then; allok = .false.; else
        ! the coordinate records: x(nx) uniform, y(ny) = yh, z(nz) uniform,
        ! float64, bit-exact vs the writer's own arithmetic
        read(iu, pos=1025, iostat=ios) cdisk
        if (ios /= 0) allok = .false.
        do ix = 1, nx
           if (cdisk(ix) /= real(ix-1,dp)*Lx/real(nx,dp)) allok = .false.
        end do
        do iy = 0, n
           if (cdisk(nx+1+iy) /= yh(iy)) allok = .false.
        end do
        do iz = 1, nz
           if (cdisk(nx+ny+iz) /= real(iz-1,dp)*Lz/real(nz,dp)) allok = .false.
        end do
        do f = 0, 2
           read(iu, pos=1024 + csh + f*ny*nx*nz*8 + 1, iostat=ios) d64
           if (ios /= 0) then; allok = .false.; exit; end if
           do iz = 1, nz; do ix = 1, nx; do iy = 1, ny
              if (d64(iy,ix,iz) /= real(fval(iy,ix,iz,f), dp)) allok = .false.
           end do; end do; end do
        end do
        close(iu)
     end if
     deallocate(d64, cdisk)
  end if

  ! (2) fp32 output -> bit-exact downcast (coords stay float64 regardless)
  call par_snap_write(f32, d, n, nx, nz, Re, t, step, 4_ip, u, v, w, yh, Lx, Lz, ok)
  if (.not. ok) allok = .false.
  if (par_rank == 0) then
     allocate(d32(ny,nx,nz))
     open(newunit=iu, file=f32, access='stream', form='unformatted', status='old', action='read', iostat=ios)
     if (ios /= 0) then; allok = .false.; else
        do f = 0, 2
           read(iu, pos=1024 + csh + f*ny*nx*nz*4 + 1, iostat=ios) d32
           if (ios /= 0) then; allok = .false.; exit; end if
           do iz = 1, nz; do ix = 1, nx; do iy = 1, ny
              if (d32(iy,ix,iz) /= real(fval(iy,ix,iz,f), sp)) allok = .false.
           end do; end do; end do
        end do
        close(iu)
     end if
     deallocate(d32)
  end if

  call par_reduce_and(allok)
  if (par_rank == 0) then
     write(*,'(a,i0,a)') ' --- test_par_snap_mp (P=', par_nproc, ') ---'
     if (allok) then; write(*,'(a)') ' ALL TESTS PASSED'
     else;            write(*,'(a)') ' SOME TESTS FAILED'; end if
  end if

  deallocate(u, v, w, hu, hv, hw)
  call par_finalize()
  if (.not. allok) error stop 1

contains
  ! unique global value at 0/1-based (iy, ix, global-iz) for field f
  real(tp) function fval(iy, ix, iz, f) result(val)
    integer(ip), intent(in) :: iy, ix, iz, f
    integer(ip) :: h
    h = modulo(iy*1009_ip + ix*2027_ip + iz*3001_ip, 100003_ip)
    val = real(h, tp)/100003.0_tp - 0.5_tp + real(f, tp)
  end function fval
end program test_par_snap_mp
