!===============================================================================
! dealias_mod.F90  --  cuFFT layer, increment 4c: 3/2-rule dealiasing helpers.
!
! For a quadratic nonlinearity, products generate modes up to ~2K; evaluating
! them on the original grid folds those high modes back onto the resolved ones
! (aliasing). The cure is the 3/2 rule: pad the spectrum to a 3N/2 grid before
! transforming to physical, form the product there, transform back, and
! truncate -- the retained modes come out alias-free.
!
! This module provides the one-sided (real-FFT / x-direction) pad and truncate.
! The real-to-complex spectrum is one-sided (modes 0..nk-1), so padding is just
! appending zeros to the high modes and truncating is dropping them -- no
! negative-frequency handling. The complex (z) direction needs a split pad
! (positive modes at the front, negative at the back); that comes with the 3D
! orchestration in spectral3d_mod.
!
! Requires -cuda on a GPU node.
!===============================================================================

module dealias_mod
  use kinds, only: tp, ip
  use cudafor
  implicit none
  private
  public :: pad_lo, trunc_lo, pad_split, trunc_split

contains

  !---------------------------------------------------------------------------
  ! Zero-pad a one-sided spectrum: cin(0:nk-1) -> cout(0:nkd-1), nkd >= nk.
  !---------------------------------------------------------------------------
  subroutine pad_lo(nk, nkd, batch, cin, cout)
    integer(ip), intent(in) :: nk, nkd, batch
    complex(tp), device     :: cin(0:nk-1, batch)
    complex(tp), device     :: cout(0:nkd-1, batch)
    integer(ip) :: k, col
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do k = 0, nkd-1
          if (k <= nk-1) then
             cout(k,col) = cin(k,col)
          else
             cout(k,col) = (0.0_tp, 0.0_tp)
          end if
       end do
    end do
  end subroutine pad_lo

  !---------------------------------------------------------------------------
  ! Truncate a one-sided spectrum: cin(0:nkd-1) -> cout(0:nk-1), keep low modes.
  !---------------------------------------------------------------------------
  subroutine trunc_lo(nkd, nk, batch, cin, cout)
    integer(ip), intent(in) :: nkd, nk, batch
    complex(tp), device     :: cin(0:nkd-1, batch)
    complex(tp), device     :: cout(0:nk-1, batch)
    integer(ip) :: k, col
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do k = 0, nk-1
          cout(k,col) = cin(k,col)
       end do
    end do
  end subroutine trunc_lo

  !---------------------------------------------------------------------------
  ! Complex (two-sided) zero-pad for the z direction. A length-nz spectrum in
  ! cuFFT order (indices 0..nz/2 = freqs 0..+nz/2, indices nz/2+1..nz-1 =
  ! freqs -(nz/2-1)..-1) is expanded to length nzd: positives to the front,
  ! negatives to the tail, zeros in the middle. Assumes the coarse Nyquist mode
  ! (index nz/2) is zero (the grid mask zeros it), so it is carried as a
  ! positive without splitting.
  !---------------------------------------------------------------------------
  subroutine pad_split(nz, nzd, batch, cin, cout)
    integer(ip), intent(in) :: nz, nzd, batch
    complex(tp), device     :: cin(0:nz-1, batch)
    complex(tp), device     :: cout(0:nzd-1, batch)
    integer(ip) :: nzh, k, col
    nzh = nz/2
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do k = 0, nzd-1
          cout(k,col) = (0.0_tp, 0.0_tp)
       end do
    end do
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do k = 0, nzh
          cout(k,col) = cin(k,col)                  ! positive freqs (+ Nyquist=0)
       end do
    end do
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do k = nzh+1, nz-1
          cout(nzd-(nz-k), col) = cin(k,col)        ! negative freqs -> tail
       end do
    end do
  end subroutine pad_split

  !---------------------------------------------------------------------------
  ! Inverse of pad_split: length-nzd spectrum -> length-nz (keep low |freq|).
  !---------------------------------------------------------------------------
  subroutine trunc_split(nzd, nz, batch, cin, cout)
    integer(ip), intent(in) :: nzd, nz, batch
    complex(tp), device     :: cin(0:nzd-1, batch)
    complex(tp), device     :: cout(0:nz-1, batch)
    integer(ip) :: nzh, k, col
    nzh = nz/2
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do k = 0, nzh
          cout(k,col) = cin(k,col)
       end do
    end do
    !$cuf kernel do(2) <<<*,*>>>
    do col = 1, batch
       do k = nzh+1, nz-1
          cout(k,col) = cin(nzd-(nz-k), col)
       end do
    end do
  end subroutine trunc_split

end module dealias_mod
