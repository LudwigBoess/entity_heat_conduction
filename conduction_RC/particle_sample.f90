! put down electrons
  do i=1,nx*ny*nz*ppg
    np_e=np_e+1
    if (np_e .gt. maxparticles) then
      write(6,*) '***** init: particle buffer overflow *****'
      call exitallpes()
    endif

    call location(rve%inf(1,np_e),rve%inf(2,np_e),rve%inf(3,np_e))
    
rnd = rand_num()
if (rnd .ge. den_fac/(1.+den_fac) ) then
   call maxwellian(real(T_1,drk),real(m_e,drk),rve%inf(4,np_e),rve%inf(5,np_e),rve%inf(6,np_e))
   rve%inf(5,np_e)=abs(rve%inf(5,np_e))
!   write(18,*) "T2"
else
   call maxwellian(real(T_perp,drk),real(m_e,drk),rve%inf(4,np_e), rve%inf(5,np_e), rve%inf(6,np_e))
   dummy_x = rve%inf(4,np_e)
   dummy_z = rve%inf(6,np_e)
   !rve%inf(np_e,5)=-abs(rve%inf(np_e,5))
   call maxwellian(real(T_2,drk),real(m_e,drk),rve%inf(4,np_e), rve%inf(5,np_e), rve%inf(6,np_e))
   rve%inf(5,np_e) = rve%inf(5,np_e) - initial_drift_factor*sqrt(2*T_2/m_e)
   do while (rve%inf(5,np_e) .gt. 0.)
      call maxwellian(real(T_perp,drk),real(m_e,drk),rve%inf(4,np_e), rve%inf(5,np_e), rve%inf(6,np_e))
      dummy_x = rve%inf(4,np_e)
      dummy_z = rve%inf(6,np_e)
      call maxwellian(real(T_2,drk),real(m_e,drk),rve%inf(4,np_e), rve%inf(5,np_e), rve%inf(6,np_e))

      rve%inf(5,np_e) = rve%inf(5,np_e) - initial_drift_factor*sqrt(2*T_2/m_e)
   enddo
   rve%inf(4,np_e) = dummy_x
   rve%inf(6,np_e) = dummy_z
endif
enddo
