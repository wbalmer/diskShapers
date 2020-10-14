;+
; NAME:   linearize_clio2
; PURPOSE:   Apply linearity correction to raw clio2 images (first detector).
; DESCRIPTION:
; CATEGORY:
; CALLING SEQUENCE:
; EXAMPLE:
; INPUTS:
; OPTIONAL INPUT PARAMETERS:
; KEYWORD INPUT PARAMETERS:
; OUTPUTS:
; KEYWORD OUTPUT PARAMETERS:
; COMMON BLOCKS:
; SIDE EFFECTS:
; RESTRICTIONS:
; PROCEDURE:
; MODIFICATION HISTORY:
;  Written 2013/04/10 by Katie Morzinski (ktmorz@arizona.edu)
;  2013/09/09 -- Accepts cubes
; BUGS/WISH LIST:
;-

FUNCTION linearize_clio2, rawimage

	im = rawimage

	;; If cube:
	if n_elements(size(im,/dim)) gt 2 then begin
		for k=0,(size(im))[3]-1 do begin
			thisim = im[*,*,k]
			;; Linearity coefficients measured on Comm2 March-April 2013
			coeff3 = [112.575 , 1.00273 , -1.40776e-06 , 4.59015e-11]
			;; Only apply to pixels above 27,000 counts.
			for n=0L,n_elements(thisim)-1 do $
				if thisim[n] gt 2.7e4 then $
						thisim[n] = coeff3[0] + coeff3[1]*thisim[n] + coeff3[2]*thisim[n]^2. + coeff3[3]*thisim[n]^3.
			im[*,*,k] = thisim
		endfor;k
	endif else begin
		;; Linearity coefficients measured on Comm2 March-April 2013
		coeff3 = [112.575 , 1.00273 , -1.40776e-06 , 4.59015e-11]
		;; Only apply to pixels above 27,000 counts.
		for n=0L,n_elements(im)-1 do $
			if im[n] gt 2.7e4 then $
					im[n] = coeff3[0] + coeff3[1]*im[n] + coeff3[2]*im[n]^2. + coeff3[3]*im[n]^3.
	endelse


RETURN,im
END
